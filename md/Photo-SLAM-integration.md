# Photo-SLAM Gaussian Pyramid 策略集成文档

> LoopSplat 项目 — 集成 [Photo-SLAM](https://arxiv.org/abs/2311.16728) (CVPR 2024) 的高斯金字塔多尺度训练策略

---

## 目录

1. [概述](#1-概述)
2. [原理：Gaussian-Pyramid-Based Learning](#2-原理gaussian-pyramid-based-learning)
3. [实现架构](#3-实现架构)
4. [使用方法](#4-使用方法)
5. [配置参数详解](#5-配置参数详解)
6. [与其他策略的协同](#6-与其他策略的协同)
7. [代码变更清单](#7-代码变更清单)

---

## 1. 概述

Photo-SLAM 的 Gaussian Pyramid 策略是一种 **从粗到细（coarse-to-fine）的渐进式训练策略**，用于增强 3DGS 建图的 photorealistic mapping 质量。

| 特性 | 说明 |
|------|------|
| **核心思想** | 训练初期用低分辨率图像监督 → 学习全局结构；后期用高分辨率 → 精细化纹理 |
| **修改对象** | **仅改变渲染分辨率 + GT 图像分辨率**，不修改高斯模型本身 |
| **作用范围** | 仅建图（Mapper），不影响跟踪（Tracker） |
| **额外开销** | 每张关键帧首次添加时构建一次金字塔（~1ms），训练时无需额外计算 |
| **论文出处** | Photo-SLAM §3.4, Figures 3(e) and 5 |

---

## 2. 原理：Gaussian-Pyramid-Based Learning

### 2.1 图像金字塔构建

```
原始图像 (1200×680) ×1.0  ←─ 全分辨率
       │
       ├── 高斯平滑 + ½下采样
       │
       ▼
  Level 1 (600×340) ×0.5   ←─ 金字塔子层 1
       │
       ├── 高斯平滑 + ½下采样
       │
       ▼
  Level 0 (300×170) ×0.25  ←─ 金字塔子层 0（最粗）
```

公式：`scale(l) = 0.5^(num_sub_levels - l)`
- `num_sub_levels = 2` → `[0.25x, 0.5x, 1.0x]`

### 2.2 渐进式训练机制（per-keyframe usage counting）

每个关键帧独立维护一个 **使用计数器**，从最粗层开始逐级前进：

```
Keyframe 刚添加时:
  counters = [8, 8]   ← 2 个子层，各可用 8 次

每次被选中训练:
  检查 counters[0] > 0? → YES → 用 Level 0 渲染，counters[0] -= 1
  (前 8 次选中: Level 0)
  ...
  检查 counters[0] > 0? → NO → 检查 counters[1] > 0? → YES → Level 1
  (接下来 8 次选中: Level 1)
  ...
  两个 counter 都归零 → 永久使用全分辨率

级别推进: [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, full, full, full, ...]
```

**关键设计意图**：
- 每个关键帧独立推进（不是全局调度），确保新加入的帧总是从粗到细
- 早期建图：大多数帧在低分辨率 → 快速学习全局结构
- 后期建图：所有帧都到了全分辨率 → 精细优化

### 2.3 深度图金字塔（无平滑）

深度图金字塔**不应用高斯平滑**，仅使用双线性插值下采样，以保留几何边缘信息：

```python
# 图像金字塔：Gaussian blur + bilinear downsample
pyr_colors = build_image_pyramid(color, num_sub_levels)

# 深度金字塔：仅 bilinear downsample（无 blur）
pyr_depths = build_depth_pyramid(depth, num_sub_levels)
```

### 2.4 渲染尺寸与相机内参

降低渲染分辨率时，相机内参**同步缩放**以保持视场角不变：

```
tanfovx = 0.5 * W / fx  → 缩放后不变（W 和 fx 同比缩放）
tanfovy = 0.5 * H / fy  → 缩放后不变
```

这使得低分辨率下渲染与全分辨率 _完全几何等价_，唯一区别是像素密度。

---

## 3. 实现架构

### 3.1 新增/修改的文件

```
src/utils/mapper_utils.py （+120 行）
├── build_image_pyramid()          → 构建图像金字塔 (Gaussian blur + downsample)
├── build_depth_pyramid()           → 构建深度金字塔 (仅 resize)
├── get_pyramid_level_dims()        → 计算金字塔级别尺寸
└── get_pyramid_render_settings()   → 生成缩放后的渲染设置

src/entities/mapper.py （+30 行，修改 map/optimize_submap）
├── Mapper.__init__                 → 解析 gaussian_pyramid 配置
├── Mapper.map()                    → 关键帧构造时构建金字塔
└── Mapper.optimize_submap()        → 训练循环按级别渲染

配置文件
├── configs/Replica/replica.yaml    → 新增 gaussian_pyramid 段
├── configs/TUM_RGBD/tum_rgbd.yaml  → 新增 gaussian_pyramid 段
└── configs/AzureKinect/azure_kinect.yaml → 新增 gaussian_pyramid 段
```

### 3.2 不修改的文件

| 文件 | 原因 |
|------|------|
| `gaussian_model.py` | 金字塔**不改变**高斯模型参数 |
| `tracker.py` | 跟踪使用全分辨率（Photo-SLAM 原文也如此） |
| `gaussian_slam.py` | Mapper 接口完全不变 |
| `utils/utils.py` | `render_gaussian_model` 接受任意尺寸的 render_settings |

### 3.3 数据流

```
Mapper.map(frame_id, ...)
│
├── 创建 keyframe dict（全分辨率 color, depth, render_settings）
│
├── [if pyramid enabled]
│   ├── build_image_pyramid(color)  → keyframe["pyramid_colors"]
│   ├── build_depth_pyramid(depth)  → keyframe["pyramid_depths"]
│   ├── get_pyramid_render_settings() × num_sub_levels → keyframe["pyramid_render_settings"]
│   └── usage_counters[frame_id] = [N_uses] × num_sub_levels
│
└── Mapper.optimize_submap(...)
    │
    └── 每轮训练:
        ├── 选择关键帧 (frame_id, keyframe)
        ├── [if pyramid enabled] 检查 usage_counters → 确定当前 level
        │   ├── level < num_sub_levels → 用 pyramid_colors[level], pyramid_render_settings[level]
        │   └── level >= num_sub_levels → 用 color, render_settings（全分辨率）
        ├── render_gaussian_model(gaussian_model, render_settings)
        ├── 计算 loss（与所选分辨率的 GT 对比）
        └── 反向传播
```

### 3.4 子图重置时的行为

当 `GaussianSLAM.start_new_submap()` 被调用时，`mapper.keyframes = []` 清空所有关键帧。由于 `_pyramid_usage_counters` 键值（frame_id）自然失效（对应的 keyframe 已被移除），无需显式清理。新子图的首帧会创建新的计数器和金字塔。

---

## 4. 使用方法

### 4.1 环境

```bash
conda activate loop_splat
```

### 4.2 命令行示例

**默认模式（不使用金字塔，与原始 LoopSplat 完全一致）：**
```bash
python run_slam.py configs/Replica/office0.yaml
```

**启用 Gaussian Pyramid：**
```bash
python run_slam.py configs/Replica/office0.yaml \
    --mapping.gaussian_pyramid.enabled true
# 或直接修改 YAML: gaussian_pyramid: { enabled: true }
```

**同时启用所有策略（IMU + GI-SLAM KF + Pyramid）：**
```bash
python run_slam.py configs/AzureKinect/144_5FPS_720p_IMU.yaml \
    --tracking.use_imu true \
    --keyframing.enable_gi_slam true \
    --mapping.gaussian_pyramid.enabled true
```

### 4.3 兼容性

| 配置 | 行为 |
|------|------|
| `gaussian_pyramid.enabled: false`（默认） | 全分辨率训练，与原始 LoopSplat 完全一致 |
| `gaussian_pyramid.enabled: true` | 渐进式多尺度训练 |
| 所有三项策略均 `false` | 等同于原始 LoopSplat |

---

## 5. 配置参数详解

```yaml
gaussian_pyramid:
  enabled: false            # 总开关
  num_sub_levels: 2         # 子分辨率层数（不含全分辨率）
  uses_per_level: 8         # 每层训练多少次后升级
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `enabled` | bool | `false` | 总开关 |
| `num_sub_levels` | int | `2` | 金字塔子层数。`2` → 3 级金字塔（0.25x + 0.5x + 1x）；`3` → 4 级（0.125x + 0.25x + 0.5x + 1x） |
| `uses_per_level` | int | `8` | 每层使用次数。总子层训练次数 = `num_sub_levels × uses_per_level`。之后自动升级到全分辨率 |

### 5.1 调参建议

| 场景 | 推荐 |
|------|------|
| 小场景（< 50 帧） | `num_sub_levels: 1`, `uses_per_level: 5` |
| 大场景 + 快速收敛 | `num_sub_levels: 2`, `uses_per_level: 5` |
| 极致质量 | `num_sub_levels: 3`, `uses_per_level: 10` |
| 接近原始行为 | `num_sub_levels: 1`, `uses_per_level: 1` |

---

## 6. 与其他策略的协同

本项目同时集成了三种策略，可独立或组合使用：

| 策略 | 配置段 | 作用范围 | 默认 |
|------|--------|---------|------|
| **IMU 损失** | `tracking.use_imu` | 跟踪 | ❌ |
| **GI-SLAM 关键帧选择** | `keyframing.enable_gi_slam` | 帧选择 | ❌ |
| **Gaussian Pyramid** | `gaussian_pyramid.enabled` | 建图 | ❌ |

### 消融实验矩阵

| 实验 | IMU | GI-KF | Pyramid | 命令 |
|------|-----|-------|---------|------|
| **基线** | ❌ | ❌ | ❌ | `python run_slam.py config.yaml` |
| **+IMU** | ✅ | ❌ | ❌ | 加 `--tracking.use_imu true` |
| **+GI-KF** | ❌ | ✅ | ❌ | 加 `--keyframing.enable_gi_slam true` |
| **+Pyramid** | ❌ | ❌ | ✅ | 加 `--mapping.gaussian_pyramid.enabled true` |
| **+IMU+GI-KF** | ✅ | ✅ | ❌ | GI-SLAM 完整 |
| **全开** | ✅ | ✅ | ✅ | 所有策略 |

### 预期效果

| 指标 | IMU | GI-KF | Pyramid |
|------|-----|-------|---------|
| ATE RMSE | ↓ 10-30% | — | — |
| 渲染 PSNR | — | ≈ | ↑ 2-5% |
| 关键帧效率 | — | ↓ 20-40% keyframes | — |
| 收敛速度 | — | — | ↑（早期更快） |

---

## 7. 代码变更清单

| 文件 | 变更 | 行数 |
|------|------|------|
| `src/utils/mapper_utils.py` | 新增 4 个金字塔工具函数 | +120 |
| `src/entities/mapper.py` | `__init__`解析配置 + `map()`构建金字塔 + `optimize_submap()`按级别渲染 | +30 |
| `configs/Replica/replica.yaml` | 新增 `gaussian_pyramid` 段 | +3 |
| `configs/TUM_RGBD/tum_rgbd.yaml` | 新增 `gaussian_pyramid` 段 | +3 |
| `configs/AzureKinect/azure_kinect.yaml` | 新增 `gaussian_pyramid` 段 | +3 |
| `md/Photo-SLAM-integration.md` | 本文档 | — |
