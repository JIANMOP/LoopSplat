# GI-SLAM 策略集成文档

> LoopSplat 项目 — 集成 [GI-SLAM](https://arxiv.org/abs/2503.18275) 的 IMU 损失函数与内容感知关键帧选择策略

---

## 目录

1. [概述](#1-概述)
2. [策略一：IMU 损失函数（已完成）](#2-策略一imu-损失函数已完成)
3. [策略二：Gaussian Visibility 关键帧选择（本次新增）](#3-策略二gaussian-visibility-关键帧选择本次新增)
4. [使用方法](#4-使用方法)
5. [配置参数详解](#5-配置参数详解)
6. [代码架构](#6-代码架构)
7. [消融实验指南](#7-消融实验指南)

---

## 1. 概述

本文档描述在 LoopSplat 框架中集成的两项 GI-SLAM 核心策略：

| 策略 | 来源论文 | 核心贡献 | 实现状态 |
|------|---------|---------|---------|
| **IMU 损失函数** | GI-SLAM §3.1 | 将 IMU 加速度/角速度约束融入跟踪优化 | ✅ 已实现 |
| **Gaussian Visibility 关键帧选择** | GI-SLAM §3.3 | 基于 3DGS 可见性覆盖的动态关键帧筛选 | ✅ 本次新增 |

---

## 2. 策略一：IMU 损失函数（已完成）

### 2.1 原理

GI-SLAM 提出将 IMU 测量值作为跟踪优化中的额外约束项，与光度损失联合优化：

```
L_total = L_photometric + L_imu

L_imu = λ_trans · L_trans + λ_rot · L_rot
```

**平移约束：** 利用线性加速度预测帧间位移：
```
Δp_imu = v_{t-1} · Δt + ½·a_t · Δt²
L_trans = ||Δp_opt - Δp_imu||²₂
```

**旋转约束：** 利用陀螺仪角速度预测帧间旋转（axis-angle）：
```
Δθ_imu = ω_t · Δt
L_rot = ||Δθ_opt - Δθ_imu||²₂
```

**速度传播：** 下一帧的初始速度由运动学积分得到：
```
v_t = v_{t-1} + a_t · Δt
```

### 2.2 实现位置

文件：`src/entities/tracker.py`

```
Tracker
├── __init__()          → 解析 use_imu, lambda_imu_trans, lambda_imu_rot
├── compute_imu_loss()  → 核心 IMU 损失计算（Eq. 5-9）
└── track()             → 在跟踪循环中获取 IMU 数据并传入 compute_losses()
```

关键代码在 [`compute_imu_loss()`](src/entities/tracker.py:137-183)：

```python
def compute_imu_loss(self, delta_pose, imu_data, dt):
    # 重力补偿
    gravity = torch.tensor([0.0, 0.0, -9.81])
    accel = accel_raw - gravity

    # 平移约束 (GI-SLAM Eq.5-6)
    imu_trans = self.prev_velocity * dt + 0.5 * accel * dt**2
    trans_loss = torch.norm(delta_trans - imu_trans, p=2)**2

    # 旋转约束 (GI-SLAM Eq.7-8)
    imu_rot = gyro * dt
    rot_loss = torch.norm(delta_rot - imu_rot, p=2)**2

    # 速度积分 (velocity propagation)
    self.prev_velocity = self.prev_velocity + accel * dt

    # 加权组合 (GI-SLAM Eq.9)
    imu_loss = self.lambda_imu_trans * trans_loss + self.lambda_imu_rot * rot_loss
```

### 2.3 依赖的 IMU 数据

| 数据集 | IMU 支持 | 实现位置 |
|--------|---------|---------|
| AzureKinect | ✅ 内置 | `datasets_azure.py:load_imu_data()` |
| TUM-RGBD | ✅ 自带 | `accelerometer.txt`（需添加读取逻辑） |
| EuRoC | ❌ 需新增 | — |
| Replica | ❌ 无 IMU | — |

---

## 3. 策略二：Gaussian Visibility 关键帧选择（本次新增）

### 3.1 原理

GI-SLAM 提出基于三个因素动态决定是否将当前帧选为建图关键帧：

**评分公式：**

```
s_i = w_covis · (1 - IoU_𝒢) + w_base · ||t_ij|| / d_med - w_mot · 𝕀(v > v_max ∨ ω > ω_max)
```

| 项 | 含义 | 高值表示 |
|----|------|---------|
| `(1 - IoU_𝒢)` | 当前帧与最近关键帧的高斯可见性差异 | 视角变化大，需要新关键帧 |
| `||t_ij|| / d_med` | 归一化平移距离 | 空间覆盖不足 |
| `-𝕀(v > v_max ∨ ω > ω_max)` | 运动模糊惩罚 | 帧模糊，不适合做关键帧 |

**决策规则：** `s_i > score_threshold` → 选为关键帧

### 3.2 实现架构

新增/修改代码：

```
src/utils/mapper_utils.py （+160 行）
├── compute_gaussian_visibility()   → 计算给定位姿下可见的高斯点 ID 集合
├── compute_gaussian_iou()          → 两组可见性 ID 集合的 IoU
├── compute_camera_velocity()       → 连续帧间的线速度/角速度估计
├── compute_median_depth()          → 中位深度（归一化基线距离）
└── gi_slam_keyframe_score()        → 整合四个子函数实现完整评分公式

src/entities/gaussian_slam.py （+140 行）
├── __init__                        → 解析 keyframing 配置段
├── _should_map_frame_gi_slam()     → 逐帧评分，决定是否选为关键帧
├── _register_gi_keyframe()         → 建图后缓存可见性+位姿
├── start_new_submap()              → 子图重置时清理 GI 状态
└── run()                           → 主循环集成动态关键帧选择
```

### 3.3 关键设计决策

#### 3.3.1 高斯可见性计算

复用已有的 `compute_frustum_point_ids()` 来判定哪些 3D 高斯点投影到当前相机视角内：

```
camera frustum → AABB 粗筛 → frustum planes 精筛 → visible Gaussian IDs
```

这避免了额外的渲染开销，仅需一次 frustum 几何检验。

#### 3.3.2 速度估计的双来源策略

| 来源 | 线速度 | 角速度 | 适用场景 |
|------|--------|--------|---------|
| 视觉里程计 | `||Δp|| / dt` | `acos(trace(R)-1)/2 / dt` | 所有数据集 |
| IMU（如可用） | — | `||gyro||`（rad/s → deg/s） | AzureKinect, TUM |

无 IMU 时角速度回退到纯视觉估计，保证所有数据集均可使用。

#### 3.3.3 时间戳感知

| 数据集 | 时间戳来源 | 回退值 |
|--------|-----------|--------|
| TUM-RGBD | `dataset.timestamps` | `1 / fps` |
| AzureKinect | `dataset.timestamps` | `1 / fps` |
| Replica | 无时间戳 | `1 / fps`（`fps=30`） |

#### 3.3.4 子图边界与 GI 状态同步

当 `start_new_submap()` 被触发（运动阈值或固定间隔），Gaussian 模型完全重建，此时：

```python
# src/entities/gaussian_slam.py @ start_new_submap()
self._gi_kf_visible_ids.clear()  # 清空旧 submap 的可见性缓存
self._gi_kf_c2ws.clear()          # 清空旧 submap 的位姿缓存
```

新 submap 的首帧自动选为关键帧（下一帧评分时仅有一个参照关键帧）。

### 3.4 性能考量

| 操作 | 触发时机 | 开销 |
|------|---------|------|
| Frustum 可见性计算 | 每帧（仅 `enable_gi_slam=True`） | ~1ms GPU |
| IoU 计算 | 非关键帧：1 次 set 操作；关键帧：0 次 | 可忽略 |
| 关键帧注册 | 每关键帧（建图后） | 等价于一次 frustum 检验 |
| 速度估计 | 每帧 | O(1) 矩阵运算 |

---

## 4. 使用方法

### 4.1 环境激活

```bash
conda activate loop_splat
```

### 4.2 配置

所有相关参数集中在一个 `keyframing` 配置段中。完整示例（以 Replica 为例）：

```yaml
# —— 控制 IMU 损失（在 tracking 段） ——
tracking:
  use_imu: false              # 启用 IMU 损失（仅 IMU 数据集）
  lambda_imu_trans: 0.01      # 平移约束权重
  lambda_imu_rot: 0.01        # 旋转约束权重

# —— 控制 GI-SLAM 关键帧选择（在 keyframing 段） ——
keyframing:
  enable_gi_slam: false       # 总开关：动态关键帧选择

  # 评分权重
  w_covis: 1.0                # 可见性差异权重
  w_base: 1.0                 # 基线距离权重
  w_mot: 2.0                  # 运动模糊惩罚权重

  # 决策阈值
  score_threshold: 0.5        # 选为关键帧的最低分数

  # 运动模糊阈值
  v_max: 0.8                  # 线速度上限 (m/s)
  omega_max: 50.0             # 角速度上限 (deg/s)

  # 其他
  min_keyframe_interval: 1    # 关键帧最小间隔（帧数）
  fps: 30.0                   # 无时间戳数据集的回退帧率
```

### 4.3 命令行启动

**基础模式（传统固定间隔关键帧）：**
```bash
python run_slam.py configs/Replica/office0.yaml
```

**启用 IMU 损失（Azure Kinect）：**
```bash
python run_slam.py configs/AzureKinect/144_5FPS_720p_IMU.yaml \
    --tracking.use_imu true
# 或修改 YAML: tracking: { use_imu: true }
```

**启用 GI-SLAM 关键帧选择：**
```bash
python run_slam.py configs/Replica/office0.yaml \
    --keyframing.enable_gi_slam true
# 或修改 YAML: keyframing: { enable_gi_slam: true }
```

**同时启用两项策略（完整 GI-SLAM 集成）：**
```bash
python run_slam.py configs/AzureKinect/144_5FPS_720p_IMU.yaml \
    --tracking.use_imu true \
    --keyframing.enable_gi_slam true \
    --keyframing.w_covis 1.0 \
    --keyframing.w_mot 2.0
```

### 4.4 兼容性说明

| 配置 | 行为 | 说明 |
|------|------|------|
| `keyframing.enable_gi_slam: false`（默认） | 固定间隔 `map_every` | **与原始 LoopSplat 完全一致** |
| `keyframing.enable_gi_slam: true` | 动态关键帧 | 基于评分在线决定 |
| `tracking.use_imu: false`（默认） | 无 IMU 损失 | 纯视觉跟踪 |
| `tracking.use_imu: true` | 带 IMU 损失 | 需数据集提供 IMU |

---

## 5. 配置参数详解

### 5.1 IMU 损失参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tracking.use_imu` | bool | `false` | 总开关。仅当数据集有 IMU 时设为 `true` |
| `tracking.lambda_imu_trans` | float | `0.01` | 平移约束权重，值越大 IMU 平移先验越强 |
| `tracking.lambda_imu_rot` | float | `0.01` | 旋转约束权重，值越大 IMU 旋转先验越强 |

### 5.2 关键帧选择参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `keyframing.enable_gi_slam` | bool | `false` | 总开关 |
| `keyframing.w_covis` | float | `1.0` | 可见性差异权重。提高 → 更密集的关键帧 |
| `keyframing.w_base` | float | `1.0` | 基线距离权重。提高 → 更多远距离关键帧 |
| `keyframing.w_mot` | float | `2.0` | 运动模糊惩罚权重。提高 → 更多拒绝模糊帧 |
| `keyframing.score_threshold` | float | `0.5` | 关键帧最低分。降低 → 更多关键帧 |
| `keyframing.v_max` | float | `0.8` | 线速度阈值 (m/s)。超过开始惩罚 |
| `keyframing.omega_max` | float | `50.0` | 角速度阈值 (deg/s)。超过开始惩罚 |
| `keyframing.min_keyframe_interval` | int | `1` | 最小关键帧间隔。`1` = 允许连续帧均为关键帧 |
| `keyframing.fps` | float | `30.0` | 无时间戳时的回退帧率 |

### 5.3 调参建议

| 场景 | 推荐调整 |
|------|---------|
| 慢速平稳运动 | `w_covis: 1.0`, `score_threshold: 0.6`（稀疏关键帧） |
| 快速大视角变化 | `w_covis: 0.5`, `score_threshold: 0.3`（密集关键帧） |
| 频繁运动模糊 | `w_mot: 3.0`, `v_max: 0.5` |
| 小型封闭场景 | `w_base: 0.5`（降低空间约束） |

---

## 6. 代码架构

### 6.1 文件依赖关系

```
gaussian_slam.py
├── imports from mapper_utils.py:
│   ├── exceeds_motion_thresholds    (已有，子图边界)
│   ├── compute_gaussian_visibility  (新增，GI-SLAM)
│   ├── compute_camera_velocity      (新增，GI-SLAM)
│   └── gi_slam_keyframe_score       (新增，GI-SLAM)
│
├── _should_map_frame_gi_slam()
│   └── 调用 mapper_utils 函数链:
│       compute_camera_velocity → compute_gaussian_visibility
│                               → gi_slam_keyframe_score
│                                   → compute_gaussian_iou
│                                   → compute_median_depth
│
├── _register_gi_keyframe()
│   └── compute_gaussian_visibility → 缓存到 _gi_kf_visible_ids
│
└── run()
    ├── [track] → [GI-SLAM keyframe eval] → [submap check] → [mapping]
    ├── [mapping 后] → _register_gi_keyframe()
    └── [每帧结尾] → 更新 _gi_prev_c2w, _gi_prev_frame_id

mapper.py
├── 不感知 GI-SLAM — 接口完全不变
└── map() 仍被 GaussianSLAM.run() 调用，只是调用时机由 GI-SLAM 决定

tracker.py
├── compute_imu_loss()    # IMU 损失（已实现，本次无改动）
├── compute_losses()      # 集成 IMU 损失到总损失
└── track()               # 跟踪循环中获取 IMU 数据
```

### 6.2 数据流

```
每帧循环 (GaussianSLAM.run):

  跟踪器输出估计位姿
       │
       ▼
  [GI-SLAM 关键帧评估]         ← 仅在 enable_gi_slam=true 时
       │                        ← 使用当前帧可见性 vs 最近关键帧可见性
       │                        ← 结合速度估计和 IMU
       ▼
  ？是否选为关键帧？
       │
  ┌────┴────┐
  │ YES     │ NO
  ▼         ▼
 建图      跳过
  │
  ▼
 [注册关键帧可见性]            ← 在 mapping 块末尾
  │                              ← 存储 visible_ids + c2w 到缓存
  ▼
 [更新速度追踪状态]             ← 每帧都执行
  │                              ← _gi_prev_c2w, _gi_prev_frame_id
  ▼
 下一帧
```

---

## 7. 消融实验指南

### 7.1 实验矩阵

| 实验编号 | IMU Loss | GI-SLAM Keyframing | 配置命令 |
|----------|---------|-------------------|---------|
| **A-基线** | ❌ | ❌ | `python run_slam.py config.yaml`（默认） |
| **B-IMU** | ✅ | ❌ | 加 `--tracking.use_imu true` |
| **C-KF** | ❌ | ✅ | 加 `--keyframing.enable_gi_slam true` |
| **D-完整** | ✅ | ✅ | 同时开启两项 |

### 7.2 评估指标

| 指标 | 来源 | 说明 |
|------|------|------|
| ATE RMSE (cm) | `output/*/ate_aligned.json` | 轨迹精度 |
| PSNR / SSIM / LPIPS | `output/*/rendering_metrics.json` | 渲染质量 |
| Depth L1 | `output/*/rendering_metrics.json` | 深度精度 |
| 关键帧数量 | 日志输出 | 策略效率 |
| 跟踪耗时/帧 | 日志输出 | 实时性 |

### 7.3 预期效果（基于 GI-SLAM 论文）

| 指标 | 预期改善（vs 基线） | 原因 |
|------|-------------------|------|
| ATE RMSE | ↓ 10-30% | IMU 约束提供额外运动先验 |
| 跟踪鲁棒性 | ↑ 运动模糊场景下成功率 | 运动惩罚过滤模糊帧 |
| 关键帧数量 | ↓ 20-40%（动态 vs 固定间隔） | 内容感知避免冗余关键帧 |
| 渲染 PSNR | ≈ 持平或略降 | 更少的关键帧可能导致细节缺失 |

---

## 附录 A：文件变更清单

| 文件 | 变更 | 行数 |
|------|------|------|
| `src/utils/mapper_utils.py` | 新增 5 个函数 | +160 |
| `src/entities/gaussian_slam.py` | 新增 2 方法 + 修改 run/start_new_submap | +140 |
| `configs/Replica/replica.yaml` | 新增 `keyframing` 段 | +10 |
| `configs/TUM_RGBD/tum_rgbd.yaml` | 新增 `keyframing` 段 | +10 |
| `configs/AzureKinect/azure_kinect.yaml` | 新增 `keyframing` 段 | +10 |

## 附录 B：Git 回退

所有变更已提交到本地 git：

```bash
# 查看提交历史
git log --oneline

# 回退到初始状态
git reset --hard HEAD~1

# 查看变更
git diff HEAD~1
```
