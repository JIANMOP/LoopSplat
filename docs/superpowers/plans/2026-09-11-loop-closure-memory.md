# 回环闭合显存优化实施计划

> **供智能体执行：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 逐项实施本计划。所有步骤使用复选框跟踪。

**目标：** 将回环闭合历史相机观测保留在 CPU，只将 GSR 最终选中的少量相机加载到 GPU，从而使约 3000 关键帧的 A4 和 Replica 长序列不再因相机深拷贝而 OOM。

**架构：** `Loop_closure` 负责以 CPU 张量保存历史观测，并为 PGO 构造不含观测的独立相机；`solver` 保持原描述子公式，但把视角选择提前到深拷贝之前；`Camera` 只在定位器真正使用观测时将 CPU RGB 原样转换到 CUDA。高斯模型、回环候选、定位损失和所有策略参数保持不变。

**技术栈：** Python 3.10、PyTorch 2.1、CUDA、Open3D、pytest、tmux。

**设计文档：** `docs/superpowers/specs/2026-09-11-loop-closure-memory-design.md`

## 全局约束

- 不修改 IMU 跟踪、高斯金字塔调度或 GI-KF 选择代码。
- 不改变关键帧数量、子图数量、回环候选、图像分辨率、GSR 迭代次数或损失函数。
- CPU RGB 必须保持原始 uint8 像素；转入 CUDA 后必须与旧实现得到相同的 float32 `[0, 1]` 张量。
- PGO 分析相机必须保留独立位姿，但不得保留 RGB、深度或梯度掩码。
- 正式运行必须使用提交后的干净工作树，并保留旧实验输出。
- A4 四种策略必须使用同一提交、同一 GPU、同一种子 0 串行运行。

---

### 任务一：相机按需加载 CPU RGB

**文件：**
- 修改：`src/gsr/camera.py:158-180`
- 创建：`tests/test_loop_closure_memory.py`

**接口：**
- 输入：`Camera.original_image` 为形状 `(3, H, W)` 的 CPU uint8 或 float32 张量。
- 输出：`Camera.load_rgb()` 将其转换为 CUDA float32 `[0, 1]`，随后按原配置计算 `grad_mask`。

- [ ] **步骤1：编写失败测试**

```python
import numpy as np
import pytest
import torch

from src.gsr.camera import Camera


def replica_gradient_config():
    return {
        "Training": {"edge_threshold": 4.0},
        "Dataset": {"type": "replica"},
    }


def make_camera(rgb, cuda_device):
    height = rgb.shape[-2] if rgb.ndim >= 2 else 32
    width = rgb.shape[-1] if rgb.ndim >= 2 else 32
    return Camera(
        0, rgb, np.ones((height, width), dtype=np.float32),
        torch.eye(4, device=cuda_device),
        torch.eye(4, device=cuda_device),
        1.0, 1.0, 0.5, 0.5, 1.0, 1.0, height, width,
        device=str(cuda_device),
    )


def test_camera_materializes_cpu_uint8_rgb_on_cuda(cuda_device):
    rgb = torch.zeros((3, 32, 32), dtype=torch.uint8)
    rgb[:, 0, :2] = torch.tensor([[0, 255], [64, 128], [255, 0]])
    camera = make_camera(rgb, cuda_device)
    camera.config = replica_gradient_config()

    camera.load_rgb()

    assert camera.original_image.is_cuda
    assert camera.original_image.dtype == torch.float32
    torch.testing.assert_close(
        camera.original_image.cpu(), rgb.float() / 255.0)
    assert camera.grad_mask.is_cuda


def test_camera_rejects_invalid_cpu_rgb_shape(cuda_device):
    camera = make_camera(torch.zeros((2, 2), dtype=torch.uint8), cuda_device)
    camera.config = replica_gradient_config()

    with pytest.raises(ValueError, match="RGB observation"):
        camera.load_rgb()
```

- [ ] **步骤2：确认测试以预期原因失败**

运行：

```bash
/home/pfy22/miniconda3/envs/loop_splat/bin/python -m pytest \
  tests/test_loop_closure_memory.py::test_camera_materializes_cpu_uint8_rgb_on_cuda \
  tests/test_loop_closure_memory.py::test_camera_rejects_invalid_cpu_rgb_shape -v
```

预期：第一个测试因为旧实现没有把内存中的 CPU RGB 移至 CUDA 而失败；第二个测试因为旧实现没有验证形状而失败。

- [ ] **步骤3：实现最小按需加载逻辑**

在 `Camera.load_rgb()` 中保留路径加载和显式 `image` 参数分支，并增加 CPU 张量分支：

```python
rgb = self.original_image
if not torch.is_tensor(rgb):
    raise TypeError("RGB observation must be a torch.Tensor")
if rgb.ndim != 3 or rgb.shape[0] != 3:
    raise ValueError("RGB observation must have shape (3, H, W)")
if rgb.dtype == torch.uint8:
    rgb = rgb.to(device=self.device, dtype=torch.float32) / 255.0
elif rgb.is_floating_point():
    rgb = rgb.to(device=self.device, dtype=torch.float32)
else:
    raise TypeError("RGB observation must be uint8 or floating point")
self.original_image = rgb
self.compute_grad_mask(self.config)
```

- [ ] **步骤4：运行任务一测试并确认通过**

运行上述两个 pytest 节点，预期 `2 passed`。

- [ ] **步骤5：提交任务一**

```bash
git add src/gsr/camera.py tests/test_loop_closure_memory.py
git commit -m "fix: materialize loop closure images on demand"
```

---

### 任务二：历史观测驻留 CPU，PGO 相机仅保留位姿

**文件：**
- 修改：`src/entities/lc.py:127-232`
- 修改：`src/entities/lc.py:314-332`
- 测试：`tests/test_loop_closure_memory.py`

**接口：**
- `Loop_closure._make_camera(...) -> Camera`：数组数据集返回 CPU uint8 CHW RGB、CPU 深度和空梯度掩码。
- `pose_only_camera_copy(camera: Camera) -> Camera`：返回位姿独立、观测字段为空的副本。

- [ ] **步骤1：编写失败测试**

```python
from types import SimpleNamespace

from src.entities.lc import Loop_closure, pose_only_camera_copy


def make_loop_closer(cuda_device):
    loop_closer = Loop_closure.__new__(Loop_closure)
    loop_closer.device = str(cuda_device)
    loop_closer.config = replica_gradient_config()
    loop_closer.proj_matrix = torch.eye(4, device=cuda_device)
    loop_closer.dataset = SimpleNamespace(
        intrinsics=np.eye(3), fovx=1.0, fovy=1.0,
        height=4, width=5,
    )
    return loop_closer


def test_loop_camera_keeps_array_observation_on_cpu(cuda_device):
    loop_closer = make_loop_closer(cuda_device)
    rgb = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(4, 5, 3)
    depth = np.ones((4, 5), dtype=np.float32)

    camera = loop_closer._make_camera(0, np.eye(4), torch.eye(4), depth, rgb)

    assert camera.original_image.device.type == "cpu"
    assert camera.original_image.dtype == torch.uint8
    assert camera.original_image.shape == (3, 4, 5)
    assert camera.grad_mask is None


def test_pose_only_camera_copy_drops_observations_and_isolates_pose(cuda_device):
    camera = make_camera(torch.zeros((3, 2, 2), dtype=torch.uint8), cuda_device)
    camera.depth = np.ones((2, 2), dtype=np.float32)
    camera.grad_mask = torch.ones((1, 2, 2), device=cuda_device)

    clone = pose_only_camera_copy(camera)
    clone.update_RT(torch.eye(3, device=cuda_device), torch.ones(3, device=cuda_device))

    assert clone.original_image is None
    assert clone.depth is None
    assert clone.grad_mask is None
    assert not torch.equal(clone.T, camera.T)
```

- [ ] **步骤2：确认测试以预期原因失败**

运行：

```bash
/home/pfy22/miniconda3/envs/loop_splat/bin/python -m pytest \
  tests/test_loop_closure_memory.py::test_loop_camera_keeps_array_observation_on_cpu \
  tests/test_loop_closure_memory.py::test_pose_only_camera_copy_drops_observations_and_isolates_pose -v
```

预期：旧 `_make_camera()` 立即创建 CUDA float32 RGB；`pose_only_camera_copy` 尚不存在。

- [ ] **步骤3：实现 CPU 驻留与位姿副本**

数组观测分支改为：

```python
rgb_cpu = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).contiguous()
cam_i.original_image = rgb_cpu
cam_i.depth = np.asarray(depth)
cam_i.grad_mask = None
```

增加模块级辅助函数并在 `construct_pose_graph()` 使用：

```python
def pose_only_camera_copy(camera):
    clone = copy.deepcopy(camera)
    clone.clean()
    return clone
```

```python
self.cam_dict[cam.uid] = pose_only_camera_copy(cam)
```

缓存中的 RGB 必须继续为 CPU 张量，缓存命中时不得调用 `.cuda()` 或提前计算梯度掩码。

- [ ] **步骤4：运行任务二和现有回环缓存测试**

```bash
/home/pfy22/miniconda3/envs/loop_splat/bin/python -m pytest \
  tests/test_loop_closure_memory.py tests/test_loop_closure_cache.py -v
```

预期：全部通过。

- [ ] **步骤5：提交任务二**

```bash
git add src/entities/lc.py tests/test_loop_closure_memory.py
git commit -m "fix: keep loop closure observations on cpu"
```

---

### 任务三：GSR 先选择视角，再复制相机

**文件：**
- 修改：`src/gsr/solver.py:110-190`
- 测试：`tests/test_loop_closure_memory.py`

**接口：**
- `select_registration_views(src_dict, tgt_dict, device=None) -> tuple[list, list]`：使用原 top-k 公式并只深拷贝选中相机。
- `gaussian_registration(...)`：继续返回现有结果字典，不改变任何键或数值流程。

- [ ] **步骤1：编写失败测试**

```python
import copy

from src.gsr.solver import select_registration_views


class CopyTrackedCamera:
    def __init__(self, uid):
        self.uid = uid
        self.copy_count = 0

    def __deepcopy__(self, memo):
        self.copy_count += 1
        return CopyTrackedCamera(self.uid)


def tracked_submap(descriptors, prefix):
    return {
        "kf_desc": torch.tensor(descriptors, dtype=torch.float32),
        "cameras": [
            CopyTrackedCamera(f"{prefix}{index}")
            for index in range(len(descriptors))
        ],
    }


def test_registration_view_selection_preserves_topk_and_limits_copies():
    source = tracked_submap([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], "s")
    target = tracked_submap([[1.0, 0.0], [0.0, 0.9], [0.7, 0.7]], "t")

    src_views, tgt_views = select_registration_views(source, target, device="cpu")

    assert [camera.uid for camera in src_views] == ["s0", "s1"]
    assert [camera.uid for camera in tgt_views] == ["t0", "t1"]
    assert sum(camera.copy_count for camera in source["cameras"]) == 2
    assert sum(camera.copy_count for camera in target["cameras"]) == 2
```

- [ ] **步骤2：确认测试以预期原因失败**

```bash
/home/pfy22/miniconda3/envs/loop_splat/bin/python -m pytest \
  tests/test_loop_closure_memory.py::test_registration_view_selection_preserves_topk_and_limits_copies -v
```

预期：`select_registration_views` 尚不存在。

- [ ] **步骤3：实现原公式的先选后复制**

```python
def select_registration_views(src_dict, tgt_dict, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src_desc = src_dict["kf_desc"].to(device)
    tgt_desc = tgt_dict["kf_desc"].to(device)
    score_cross = torch.einsum("id,jd->ij", src_desc, tgt_desc)
    score_best_src, _ = score_cross.topk(1)
    _, src_indices = score_best_src.view(-1).topk(min(2, score_best_src.numel()))
    score_best_tgt, _ = score_cross.T.topk(1)
    _, tgt_indices = score_best_tgt.view(-1).topk(min(2, score_best_tgt.numel()))
    src_views = [copy.deepcopy(src_dict["cameras"][i.item()]) for i in src_indices]
    tgt_views = [copy.deepcopy(tgt_dict["cameras"][i.item()]) for i in tgt_indices]
    return src_views, tgt_views
```

`gaussian_registration()` 调用该函数，并删除对完整 `src_dict['cameras']` 和 `tgt_dict['cameras']` 的深拷贝。高斯模型深拷贝和后续定位代码保持原样。

- [ ] **步骤4：运行全部显存回归测试**

```bash
/home/pfy22/miniconda3/envs/loop_splat/bin/python -m pytest \
  tests/test_loop_closure_memory.py tests/test_loop_closure_cache.py -v
```

预期：全部通过，且视角 UID 与旧公式一致。

- [ ] **步骤5：提交任务三**

```bash
git add src/gsr/solver.py tests/test_loop_closure_memory.py
git commit -m "fix: copy only selected registration cameras"
```

---

### 任务四：本机回归与 GPU 冒烟验证

**文件：**
- 不新增生产文件
- 验证：`src/`、`scripts/`、`tests/`

**接口：**
- 输入：修改后的工作树。
- 输出：编译、完整测试、策略配置检查和 GPU 冒烟测试证据。

- [ ] **步骤1：运行编译检查**

```bash
/home/pfy22/miniconda3/envs/loop_splat/bin/python -m compileall -q \
  src scripts run_slam.py run_slam_azure.py
```

预期：退出码 0，无语法错误。

- [ ] **步骤2：运行完整测试套件**

```bash
/home/pfy22/miniconda3/envs/loop_splat/bin/python -m pytest -q
```

预期：零失败。

- [ ] **步骤3：确认三个策略代码未被修改**

```bash
git diff pre-lc-memory-fix-20260911 -- \
  src/entities/tracker.py src/entities/mapper.py \
  src/utils/mapper_utils.py src/entities/gaussian_slam.py
```

预期：无差异。

- [ ] **步骤4：运行本机 TUM GPU 冒烟测试**

```bash
DISABLE_WANDB=true /home/pfy22/miniconda3/envs/loop_splat/bin/python \
  run_slam.py configs/smoke/tum_baseline.yaml
```

预期：完成跟踪、建图和正式输出阶段，无 CUDA OOM 或 Traceback。

- [ ] **步骤5：检查差异并提交必要的验证文档更新**

```bash
git diff --check
git status --short
```

只提交本计划直接产生的文件，不修改或清理用户的其他文件。

---

### 任务五：推送代码并在服务器验证和运行 A4

**文件：**
- 本机：已提交的源代码和测试
- 服务器仓库：`/root/autodl-tmp/LoopSplat`
- 验证输出：`/root/autodl-fs/output/validation/lc-memory-equivalence`
- 正式输出：`/root/autodl-fs/output/ablation/A4_*`

**接口：**
- 输入：本机已验证且提交的源代码。
- 输出：同版本服务器代码、短场景等价性结果、A4 四项串行 tmux 任务。

- [ ] **步骤1：推送实现提交**

```bash
git push origin main
```

- [ ] **步骤2：服务器同步并验证环境**

```bash
source /etc/network_turbo
cd /root/autodl-tmp/LoopSplat
git pull --ff-only
git status --short
git rev-parse HEAD
```

预期：工作树干净，提交与本机一致。

- [ ] **步骤3：运行短场景等价性检查**

将 `LOOPSPLAT_OUTPUT_ROOT` 指向独立验证目录后运行 `A1_0 seed 0`，不得写入正式 `ablation/A1_0`：

```bash
LOOPSPLAT_OUTPUT_ROOT=/root/autodl-fs/output/validation/lc-memory-equivalence \
python scripts/run_ablation.py --experiment A1_0 --seeds 0
```

对比旧正式 A1_0 的回环边、ATE、PSNR、SSIM、LPIPS 和深度指标。允许确定性 GPU 浮点误差，但不得出现回环候选或评价协议变化。

- [ ] **步骤4：创建前台可见的 A4 tmux 串行任务**

在 `ablation_A4_memory_fix` 会话中依次执行：

```bash
python scripts/run_ablation.py --experiment A4_0 --seeds 0
python scripts/run_ablation.py --experiment A4_1 --seeds 0
python scripts/run_ablation.py --experiment A4_2 --seeds 0
python scripts/run_ablation.py --experiment A4_3 --seeds 0
```

tmux 内不得使用 `nohup` 或将 Python 命令放入后台；控制器输出同时写入 `/root/autodl-fs/output/controller_logs/ablation-A4-memory-fix-seed0.log`。

- [ ] **步骤5：启动后验证**

```bash
tmux ls
tmux capture-pane -pt ablation_A4_memory_fix:0.0 -S -80
ps -eo pid,etimes,cmd | grep -E '[r]un_ablation.py|[r]un_slam.py'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

预期：只有一个 SLAM 子进程使用 GPU，日志显示首个 A4 实验进入跟踪或建图。

- [ ] **步骤6：A4 完成后的正式审计**

确认四项 `status.json` 均为 `succeeded`、`formal_outputs_complete()` 均为真、没有 OOM/磁盘写满/Traceback，并记录每项峰值显存与运行时间。只有这些证据齐全后才能宣布修复成功。
