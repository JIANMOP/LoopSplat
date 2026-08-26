# LoopSplat 实验与消融指南

本文档对应当前代码中的三项增量策略：IMU 跟踪约束、Gaussian Pyramid（高斯金字塔）和 GI-KF（关键帧选择）。正式实验只使用 TUM RGB-D、Replica 和 FMDataset；自采 Azure Kinect 序列只用于定性展示和工程验证，不进入消融表。

## 1. 投稿实验边界

### 1.1 正式实验矩阵

| 组 | 数据集 | 场景数 | 策略数 | 随机种子 | 正式运行数 | 轨迹 GT |
|---|---:|---:|---:|---:|---:|---:|
| A | TUM RGB-D | 5 | 4 | 3 | 60 | 有 |
| R | Replica | 8 | 4 | 3 | 96 | 有 |
| C | FMDataset | 3 | 6 | 3 | 54 | 无 |
| 合计 | 3 个数据集 | 16 | 70 个配置 | 3 | 210 | — |

每个配置固定运行 `seed=0,1,2`，结果报告均值和样本标准差。不要用单次运行替代三随机种子正式结果。

### 1.2 策略编号

TUM 和 Replica 没有 IMU，使用相同的四策略：

| 后缀 | 名称 | GI-KF | Pyramid | IMU |
|---|---|---:|---:|---:|
| `_0` | Baseline | 关 | 关 | 关 |
| `_1` | +GI-KF | 开 | 关 | 关 |
| `_2` | +Pyramid | 关 | 开 | 关 |
| `_3` | +KF+Pyramid | 开 | 开 | 关 |

FMDataset 有相机—IMU标定，使用六策略：

| 后缀 | 名称 | GI-KF | Pyramid | IMU |
|---|---|---:|---:|---:|
| `_0` | Baseline | 关 | 关 | 关 |
| `_1` | +IMU | 关 | 关 | 弱旋转先验 |
| `_2` | +GI-KF | 开 | 关 | 关 |
| `_3` | +Pyramid | 关 | 开 | 关 |
| `_4` | +KF+Pyramid | 开 | 开 | 关 |
| `_5` | +ALL | 开 | 开 | 弱旋转先验 |

正式 IMU 策略只使用陀螺仪旋转约束：`lambda_imu_rot=0.001`、`lambda_imu_trans=0.0`。当前 FMDataset 没有可靠的逐序列加速度计 bias、长静止初始化窗口和轨迹 GT，直接启用双重积分平移约束会把不确定的速度与重力状态写入正式结论，因此不使用旧的旋转+平移配置。`+IMU` 和 `+ALL` 使用完全相同的 IMU 权重。

FMDataset 的所有 C 组配置（包括 Baseline）固定使用单 OpenMP 线程的 CPU Open3D 视觉里程计（`odometer_device=cpu`、`odometer_omp_threads=1`）。GPU Open3D 以及多线程 CPU Open3D 在相同输入上存在不可忽略的重复运行差异，会把里程计随机性混入策略消融；正式运行器会在启动 C 组子进程前设置 `OMP_NUM_THREADS=1`，GPU 仍用于后续跟踪、渲染和建图。GI-KF 的高运动保护使用 `high_motion_max_gap=1`：线速度超过 `0.8 m/s` 或角速度超过 `50 deg/s` 时，在首个满足有效深度和最小间隔条件的帧立即选为关键帧，不再让运动惩罚连续跳过三个高运动帧。

### 1.3 场景编号

| 组 | 编号与场景 |
|---|---|
| A | A1 `fr1/desk`；A2 `fr1/desk2`；A3 `fr1/room`；A4 `fr2/xyz`；A5 `fr3/long_office_household` |
| R | R1–R5 `office0`–`office4`；R6–R8 `room0`–`room2` |
| C | C1 `dorm1_fast1`；C2 `dorm2_fast`；C3 `hotel_fast1` |

### 1.4 Azure 的定位

Azure Kinect 序列是自采数据，可在论文实验设置或定性结果中简要说明，用于证明系统能够处理真实传感器数据。它不参与正式消融，原因如下：

- 没有真值位姿，不能报告 ATE/RPE；
- 当前 IMU 约为 5 Hz，且数据中没有可靠的相机—IMU外参，不能把它当作 IMU 策略的严谨验证；
- RGB 与 depth 来自不同相机，必须先用双相机内参、畸变和 `T_color_depth` 做几何注册，不能只缩放图像。

正式的 IMU 消融仅在 FMDataset 上完成。旧的 `rergb` 单场景配置因路径失效且不满足几何配准要求已移除；论文与冒烟统一使用经过标定配准的 `configs/AzureKinect/144_5FPS_720p_IMU.yaml`。

## 2. 数据与环境检查

从仓库根目录执行：

```bash
conda activate loop_splat
cd /root/autodl-tmp/LoopSplat
export LOOPSPLAT_OUTPUT_ROOT=/root/autodl-fs/output/ablation
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
git status --short
git rev-parse HEAD
```

正式运行要求：CUDA 可用、Git 工作区为空、所有实验使用相同的实验代码指纹和 GPU/CUDA 环境。Git commit 仍写入 `manifest.json` 供追溯，但文档提交不同不会阻止续跑或汇总；工作区不干净时仍会拒绝正式运行。

数据应位于：

```text
data/TUM_RGBD-SLAM/rgbd_dataset_freiburg1_desk/
data/Replica/office0/results/             # frame*.jpg + depth*.png
data/Replica/office0/traj.txt
data/FMDataset/dorm1/dorm1_fast1/
data/AzureKinect/144_5FPS_720p_IMU/
```

Replica 八个场景均应有 2000 对 RGB-D 图像和 2000 行轨迹：

```bash
for scene in office0 office1 office2 office3 office4 room0 room1 room2; do
  printf '%-8s rgb=%s depth=%s traj=%s\n' "$scene" \
    "$(find "data/Replica/$scene/results" -maxdepth 1 -name 'frame*.jpg' | wc -l)" \
    "$(find "data/Replica/$scene/results" -maxdepth 1 -name 'depth*.png' | wc -l)" \
    "$(wc -l < "data/Replica/$scene/traj.txt")"
done
```

Azure 应包含 `color/`、`depth/`、`frame_info.json`、`imu.txt` 和 `camera_parameters.json`。第一次运行会创建 `processed_images/redepth/`；其中 `processing_metadata.json` 记录配准所用标定，标定或预处理方式变化时缓存会自动重建。

先验证代码：

```bash
python -m pytest -q
python -m compileall -q src scripts run_slam.py run_slam_azure.py
```

## 3. 正式运行前的 GPU 冒烟

冒烟配置只处理 6–8 帧，用来验证代码路径和输出契约，不能用于论文表格。

```bash
export DISABLE_WANDB=true

# FM：分别验证基线、IMU、Pyramid 和 GI-KF
export OMP_NUM_THREADS=1
python run_slam.py configs/smoke/fm_baseline.yaml
python run_slam.py configs/smoke/fm_imu.yaml
python run_slam.py configs/smoke/fm_pyramid.yaml
python run_slam.py configs/smoke/fm_keyframing.yaml
unset OMP_NUM_THREADS

# 有 GT 的公开数据
python run_slam.py configs/smoke/tum_baseline.yaml
python run_slam.py configs/smoke/replica_baseline.yaml

# 自采 Azure：只验证标定 RGB-D 链路，不开 IMU
python run_slam_azure.py configs/smoke/azure_baseline.yaml
```

一次合格的冒烟至少应满足：

- 进程退出码为 0，`status.json` 的 `state` 为 `succeeded`；
- `rendering_metrics_observed_view.json` 中 PSNR/SSIM/LPIPS/Depth-L1 均为有限值；
- TUM/Replica 的 `trajectory_status.json` 为 `available`，并生成 ATE/RPE；
- Azure 的轨迹状态应为 `skipped_no_ground_truth`，这是预期结果；
- IMU 冒烟的 `imu_tracking_summary.yaml` 至少有一个 `valid: true` 的预测；
- Pyramid 冒烟的 `gaussian_pyramid_summary.yaml` 中 `enabled: true` 且 `optimizer_step_count > 0`；
- GI-KF 冒烟生成非空的 `keyframe_decisions.jsonl`。

冒烟结果位于：

```text
output/smoke/<scene_name>/seed_<seed>/<UTC时间>_<随机后缀>/
```

## 4. 生成和运行正式矩阵

### 4.1 预览，不启动 SLAM

```bash
python scripts/run_ablation.py --dry-run
```

正确输出应为 `210` 个作业，即 `70 configurations × 3 seeds`。分组预览：

```bash
python scripts/run_ablation.py --dry-run --group A   # 20×3 = 60
python scripts/run_ablation.py --dry-run --group R   # 32×3 = 96
python scripts/run_ablation.py --dry-run --group C   # 18×3 = 54
```

### 4.2 先完成代表性单配置

每条命令默认运行三个种子：

```bash
python scripts/run_ablation.py --experiment A1_0
python scripts/run_ablation.py --experiment R1_0
python scripts/run_ablation.py --experiment C1_5
```

如果只想诊断某个种子，可以临时使用 `--seeds 0`，但该结果不能被正式汇总器当成完整实验：

```bash
python scripts/run_ablation.py --experiment C1_1 --seeds 0
```

### 4.3 分组或全量运行

推荐按组运行，便于发现数据或显存问题：

```bash
python scripts/run_ablation.py --group A --seeds 0 1 2
python scripts/run_ablation.py --group R --seeds 0 1 2
python scripts/run_ablation.py --group C --seeds 0 1 2
```

也可以一次提交全部 210 次：

```bash
python scripts/run_ablation.py --seeds 0 1 2
```

后台运行示例：

```bash
mkdir -p logs
nohup python scripts/run_ablation.py --group R --seeds 0 1 2 \
  > logs/ablation-replica.log 2>&1 &
tail -f logs/ablation-replica.log
```

不要预估固定“几小时跑完”。先记录一个完整场景各策略三个种子的实测时间，再按剩余配置数估算；场景帧数、关键帧数量和闭环注册都会显著影响耗时。

### 4.4 中断续跑与强制重跑

重复执行同一命令会自动跳过“种子、完整配置哈希、实验代码指纹均一致”的成功结果：

```bash
python scripts/run_ablation.py --group R --seeds 0 1 2
```

组内只要有一个任务失败，命令最终退出码就是非零；已经成功并且输出完整的任务仍会在下次执行相同命令时跳过，失败任务会重新运行。不能只根据外层 tmux/nohup 命令结束判断整组成功，必须同时检查末尾 `SUMMARY` 和各目录的 `status.json`。

`--force` 会创建新的带时间戳目录，不会覆盖旧结果：

```bash
python scripts/run_ablation.py --experiment R1_0 --seeds 0 --force
```

只修改并提交 `md/`、`docs/` 等文档不会改变实验代码指纹，已有成功结果仍会续用。修改 `src/`、`scripts/`、`configs/`、`run_slam.py`、`run_slam_azure.py`，或改变配置、随机种子时，运行器不会把旧结果误认为可续跑结果。正式矩阵开始后不要中途修改实验代码或配置；如必须修复，应重新运行所有受影响的对比项。

## 5. 输出文件与真实性审计

正式输出根目录由 `LOOPSPLAT_OUTPUT_ROOT` 控制；未设置时默认使用仓库内的 `output/ablation`。服务器统一设置为 `/root/autodl-fs/output/ablation`，其目录结构为：

```text
/root/autodl-fs/output/ablation/<experiment_id>/seed_<seed>/<UTC时间>_<随机后缀>/
```

核心文件：

| 文件 | 用途 |
|---|---|
| `config.input.yaml` | 消融运行器生成的输入配置 |
| `config.yaml` | SLAM 实际保存的最终配置 |
| `manifest.json` | Git 提交、实验代码指纹、配置哈希、命令、GPU/CUDA、请求/生效策略 |
| `status.json` | 成功/失败、帧数、关键帧数、子图数、耗时、显存峰值 |
| `run_statistics.yaml` | 关键帧/子图/耗时/显存明细，以及视觉里程计冻结次数、原因和帧号；C 组以 CPU 为主设备，因此正常情况下 CPU 二次回退次数为 0 |
| `run.log` | 完整标准输出和错误日志 |
| `effective_features.yaml` | 实际启用的 IMU/Pyramid/GI-KF |
| `imu_tracking_summary.yaml` | IMU 样本、预测有效性、提交次数，以及最佳位姿上的旋转/平移残差和实际加权 loss |
| `gaussian_pyramid_summary.yaml` | Pyramid 是否生效及优化步数 |
| `keyframe_decisions.jsonl` | GI-KF 每帧的选择记录，仅 GI-KF 开启时要求 |
| `evaluation_protocol.json` | 固定评估帧、地图来源、是否全局细化 |
| `evaluation_frame_ids.json` | 固定观测视角帧号 |
| `rendering_metrics_observed_view.json` | PSNR/SSIM/LPIPS/Depth-L1 |
| `trajectory_status.json` | 轨迹指标可用或无 GT 跳过 |
| `ate_aligned.json`、`rpe.json`、`trajectory_metrics.json` | 有 GT 数据的 ATE/RPE |
| `global_refinement_status.json` | 明确记录正式评估未做额外全局细化 |

### 5.1 按需导出全局高斯 PLY

正式消融默认关闭 `evaluation.run_reconstruction` 和全局高斯精修，以免不同策略额外承担重建或优化预算。因此默认不会生成 TSDF 网格 `mesh/cleaned_mesh.ply`，也不会生成精修后的 `*_global_splats.ply`。渲染评估使用的是从 `submaps/*.ckpt` 重新加载并直接拼接的未精修全局高斯；这些 checkpoint 会保留在运行目录中，所以实验结束、GPU 内存释放后仍可按需导出：

```bash
RUN_DIR=/root/autodl-fs/output/ablation/A1_0/seed_0/<run-dir>
python scripts/export_gaussian_ply.py "$RUN_DIR"
```

默认输出：

```text
<run-dir>/unrefined_global_splats.ply
```

指定其他输出位置：

```bash
python scripts/export_gaussian_ply.py "$RUN_DIR" \
  --output /root/autodl-fs/output/visualization/A1_0_seed0.ply
```

若目标文件已经存在，脚本会拒绝覆盖；确认需要替换时使用 `--force`。导出需要 CUDA GPU，PLY 保存的是带位置、球谐颜色、不透明度、尺度和旋转属性的 3D Gaussian Splat，不是三角网格，应使用支持 3DGS PLY 的查看器。该文件仅用于定性检查和论文可视化，不属于正式结果完整性条件，不影响续跑判定和指标汇总。一般只为代表性场景或 seed 0 导出，避免 210 次正式运行产生大量重复文件。

状态为 `failed` 的旧运行即使 checkpoint 齐全，也只能导出用于故障诊断，不能手动改成 `succeeded` 或与修复后的正式结果混合。修复实验代码后源指纹会改变，必须重新运行相应 seed。

审计时不要只看 `effective_features.yaml`。正式运行的完整性检查还会验证：

- `manifest.json` 中 `git_dirty=false`，Git 提交、实验代码指纹、GPU、CUDA 和配置哈希存在；
- 请求策略与实际策略完全一致；
- 评估帧号非空、唯一、有序，manifest 与保存文件一致；
- 正式地图统一来自未额外细化的全局高斯拼接；
- IMU 开启时至少一条有效预测，且数据异常行没有超过阈值；
- Pyramid 开启时确实执行了优化步；
- GI-KF 开启时确实记录了关键帧决策；
- 指标有限且深度有效像素数大于 0；
- 有 GT 时 ATE/RPE 文件齐全，无 GT 时状态明确为跳过。

检查单个运行目录：

```bash
python -c "from src.utils.experiment_utils import formal_outputs_complete; print(formal_outputs_complete('/root/autodl-fs/output/ablation/R1_0/seed_0/<run-dir>'))"
```

输出必须为 `True`。把 `<run-dir>` 替换为实际目录名。

## 6. 汇总结果

终端检查：

```bash
python scripts/aggregate_results.py --format terminal
```

生成论文表格草稿和机器可读结果：

```bash
REPORT_ROOT="${LOOPSPLAT_OUTPUT_ROOT:-$PWD/output/ablation}"
mkdir -p "$REPORT_ROOT"
python scripts/aggregate_results.py --format markdown > "$REPORT_ROOT/results.md"
python scripts/aggregate_results.py --format json > "$REPORT_ROOT/results.json"
```

汇总器只接受每个配置恰好包含 `seed=0,1,2` 的成功结果，并检查三个种子的实验代码指纹、GSR 预算、GPU/CUDA一致；同一场景的所有策略还必须使用兼容的评估协议。Git commit 可以因纯文档提交而不同。若某配置完全未运行，会显示缺失；若只有 1–2 个种子，或混入不同实验代码/环境，会直接报错，不能静默生成不严谨表格。

实验代码指纹包含 `src/`、`scripts/`、`configs/` 及两个 SLAM 入口。本次输出目录改造会开启一个新的指纹版本：在该提交之前生成的正式结果不会被自动续跑识别，也不能与新结果混合汇总。正式矩阵尚未开始时直接以本提交为起点；若已有旧正式结果，则至少应完整重跑同一场景的全部对比策略和三个种子，禁止只补跑单个配置后混用。

## 7. 消融比较方法

### 7.1 TUM 与 Replica

每个场景分别计算：

| 对比 | 结论范围 |
|---|---|
| `_1 - _0` | GI-KF 的独立贡献 |
| `_2 - _0` | Pyramid 的独立贡献 |
| `_3 - _0` | 两策略联合效果 |
| `_3 - _1` | 在 GI-KF 上增加 Pyramid 的增量 |
| `_3 - _2` | 在 Pyramid 上增加 GI-KF 的增量 |

TUM 与 Replica 都有 GT，主表应包含 ATE、平移/旋转 RPE，以及固定观测视角的 PSNR、SSIM、LPIPS、Depth-L1。还要同时报告关键帧数、子图数、SLAM 时间和峰值显存，避免只论质量、不论代价。

Replica 当前数据没有 Co-SLAM 风格的 `gt_mesh_cull_virt_cams.ply` 和 `gt_pc_unseen.npy`，因此不要伪造或套用依赖这些文件的 mesh reconstruction 指标。当前严谨可报告的是轨迹、固定观测视角渲染/深度和效率指标。

### 7.2 FMDataset

每个场景分别计算：

| 对比 | 结论范围 |
|---|---|
| `_1 - _0` | IMU 的独立贡献 |
| `_2 - _0` | GI-KF 的独立贡献 |
| `_3 - _0` | Pyramid 的独立贡献 |
| `_4 - _0` | GI-KF + Pyramid 的联合效果 |
| `_5 - _4` | 在 GI-KF + Pyramid 基础上增加 IMU 的边际贡献，核心对比 |
| `_5 - _1` | 在 IMU 基础上增加 GI-KF + Pyramid 的边际贡献 |

FMDataset 没有轨迹 GT，所以不能报告 ATE/RPE，也不能把同帧固定观测视角指标称为 novel-view synthesis。它用于验证快速运动下三种策略对渲染/深度质量和效率的影响，并提供 IMU 策略的公开数据证据。

### 7.3 统计和表述原则

- 表格使用 `mean ± std`，明确 `n=3`；
- 同一场景内比较策略，不直接用不同场景的绝对指标证明策略优劣；
- 同时报告提升和退化，不筛选“最好看的”种子；
- 不预先写“应提升 2–5%”之类结果，结论必须由正式数据决定；
- 若某策略只在部分场景有效，应表述为条件性收益并分析失败场景；
- 冒烟结果、自采 Azure 和正式三种子结果必须分开存放和描述。

## 8. 论文建议表格

最低限度建议准备：

1. TUM：5 场景 × 4 策略，轨迹、固定观测视角和效率；
2. Replica：8 场景 × 4 策略，同上；
3. FMDataset：3 场景 × 6 策略，固定观测视角和效率；
4. 三张单因素/增量对比图：IMU、Pyramid、GI-KF；
5. Azure 自采序列的一组轨迹/重建可视化，仅作定性展示；
6. 失败案例或退化场景分析。

以毕业所需的较低档期刊为目标，这套数据体量足以开始写方法、相关工作和实验设置，但“可以投稿”仍以 210 次正式运行全部完成、审计通过、趋势可解释为前提。代码冒烟通过不等于实验结论已经成立。

## 9. 服务器同步流程

本机修改提交并推送后，在服务器执行：

```bash
cd /root/autodl-tmp/LoopSplat
source /etc/network_turbo
git pull
conda activate loop_splat
mkdir -p /root/autodl-fs/output/ablation
grep -qxF 'export LOOPSPLAT_OUTPUT_ROOT=/root/autodl-fs/output/ablation' /root/.bashrc || \
  echo 'export LOOPSPLAT_OUTPUT_ROOT=/root/autodl-fs/output/ablation' >> /root/.bashrc
source /root/.bashrc
echo "$LOOPSPLAT_OUTPUT_ROOT"
git rev-parse HEAD
git status --short
python -m pytest -q
```

确认环境变量输出为 `/root/autodl-fs/output/ablation`、服务器包含相同实验代码、工作区为空且数据路径存在后，先重复第 3 节 GPU 冒烟，再启动正式矩阵。正式运行器和汇总器会读取同一个输出根目录；冒烟结果仍保存在仓库的 `output/smoke`。纯文档提交不会影响续跑；不要在服务器上直接编辑正式配置或算法代码，否则实验代码指纹会变化。

## 10. 故障处理

- `formal runs require a clean Git worktree`：提交或明确处理本地修改后再运行，不要绕过检查；
- 汇总报 seeds 错误：补齐同一实验代码指纹下的 0、1、2 三个种子；
- 汇总报 mixed source fingerprint/hardware/protocol：不能把这些结果放进同一统计组，应统一实验代码与环境后重跑受影响配置；
- IMU 完整性失败：检查 `imu_tracking_summary.yaml` 的时间覆盖、丢弃行和 `valid_prediction_count`；
- Pyramid 完整性失败：检查 `enabled` 与 `optimizer_step_count`，不能只看 YAML 开关；
- FMDataset 出现 `Singular 6x6 linear system`：先查看 `run_statistics.yaml` 的 `visual_odometry`。C 组固定由 CPU 求解，并统一使用 0.5 m/帧、60°/帧上限；CPU 求解失败或越界时冻结一帧。若 `identity_fallback_count` 持续增加或轨迹仍有大跳变，不应把该场景作为成功结果；
- Azure 缓存异常：检查 `processed_images/redepth/processing_metadata.json`；标定或预处理配置变化后，加载器应自动重建缓存；
- CUDA OOM：先确认当前代码已使用受限单 GPU FAISS 临时区；仍失败时记录配置与峰值并检查高斯数量。降低并发或只对回环重叠估计做统一的确定性采样，不得单独降低某个消融策略的优化预算。
