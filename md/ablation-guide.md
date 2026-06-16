# 消融实验指南

> LoopSplat — GI-SLAM + Photo-SLAM 消融实验完整运行流程

---

## 1. 环境准备

```bash
conda activate loop_splat
cd /data/p/.pfy/LoopSplat
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 2. 实验设计

### 三个数据集组

| 组 | 数据集 | IMU | GT 位姿 | 评估指标 |
|---|--------|-----|---------|---------|
| **A** | TUM RGB-D（5个场景） | ❌ | ✅ | ATE + PSNR + SSIM + LPIPS + Depth L1 |
| **B** | AzureKinect（1个场景） | ✅ | ✅ | ATE + PSNR + SSIM + LPIPS + Depth L1 |
| **C** | FMDataset（3个场景） | ✅ | ❌ | PSNR + SSIM + LPIPS + Depth L1 |

### 策略编号

**A 组（无 IMU，2 策略 → 4 组合）：**

| 编号 | 名称 | GI-KF | Pyramid |
|------|------|-------|---------|
| `_0` | Baseline | ❌ | ❌ |
| `_1` | +GI-KF | ✅ | ❌ |
| `_2` | +Pyramid | ❌ | ✅ |
| `_3` | +KF+Pyramid | ✅ | ✅ |

**B / C 组（有 IMU，3 策略 → 6 组合）：**

| 编号 | 名称 | IMU | GI-KF | Pyramid |
|------|------|-----|-------|---------|
| `_0` | Baseline | ❌ | ❌ | ❌ |
| `_1` | +IMU | ✅ | ❌ | ❌ |
| `_2` | +GI-KF | ❌ | ✅ | ❌ |
| `_3` | +Pyramid | ❌ | ❌ | ✅ |
| `_4` | +KF+Pyramid | ❌ | ✅ | ✅ |
| `_5` | +ALL | ✅ | ✅ | ✅ |

### 场景清单

**A 组 — TUM RGB-D（5 场景 × 4 策略 = 20 实验）：**

| 编号 | 场景 | 帧数 | 描述 |
|------|------|------|------|
| A1 | freiburg1_desk | ~592 | 桌面场景 |
| A2 | freiburg1_desk2 | ~600 | 桌面场景2 |
| A3 | freiburg1_room | ~1300 | 房间 |
| A4 | freiburg2_xyz | ~3600 | 大范围运动 |
| A5 | freiburg3_long | ~2500 | 长走廊 |

**B 组 — AzureKinect（1 场景 × 6 策略 = 6 实验）：**

| 编号 | 场景 | 描述 |
|------|------|------|
| B1 | 144_5FPS_720p_IMU | 自采集，含 IMU |

**C 组 — FMDataset（3 场景 × 6 策略 = 18 实验）：**

FMDataset 是 RGB-D + IMU 数据集，由 Azure Kinect DK 在真实室内环境中采集。**无真值位姿**，因此 C 组评估仅基于渲染指标（PSNR / SSIM / LPIPS / Depth L1），不计算 ATE。

| 编号 | 场景 | 帧数 | IMU 样本 | 时长 | 场景特征 |
|------|------|------|---------|------|---------|
| C1 | dorm1_fast1 | 869 | 5961 | ~29s | 学生宿舍，快速移动，桌椅/床铺/杂物，纹理丰富 |
| C2 | dorm2_fast | 823 | 5658 | ~27s | 学生宿舍 2 号，快速移动，布局不同，纹理丰富 |
| C3 | hotel_fast1 | 648 | 4488 | ~22s | 酒店房间，快速移动，家具/窗帘/地毯，纹理中等 |

FM 数据集的三场景均选择 **fast 速度**（相机移动较快），这样：
- 更能体现 **IMU 损失**在快速运动下的跟踪鲁棒性优势（视觉跟踪在快速运动时容易丢失）
- 更能体现 **GI-KF 关键帧选择**的运动模糊过滤效果
- **Pyramid** 在纹理丰富的宿舍场景可能提升 PSNR

**C 组完整实验清单：**

| 实验 ID | 场景 | 策略 | IMU | GI-KF | Pyramid | 验证目标 |
|---------|------|------|-----|-------|---------|---------|
| **C1_0** | dorm1_fast1 | Baseline | ❌ | ❌ | ❌ | 快速运动下的视觉跟踪基线 |
| **C1_1** | dorm1_fast1 | +IMU | ✅ | ❌ | ❌ | **IMU 单独贡献（快速运动下最关键）** |
| **C1_2** | dorm1_fast1 | +GI-KF | ❌ | ✅ | ❌ | GI-KF 在富纹理场景的关键帧效率 |
| **C1_3** | dorm1_fast1 | +Pyramid | ❌ | ❌ | ✅ | Pyramid 在富纹理场景的渲染提升 |
| **C1_4** | dorm1_fast1 | +KF+Pyramid | ❌ | ✅ | ✅ | 两个非 IMU 策略的组合 |
| **C1_5** | dorm1_fast1 | +ALL | ✅ | ✅ | ✅ | 三策略全开，与 B1_5 跨数据集对比 |
| **C2_0** | dorm2_fast | Baseline | ❌ | ❌ | ❌ | 不同宿舍场景的基线 |
| **C2_1** | dorm2_fast | +IMU | ✅ | ❌ | ❌ | IMU 在不同场景的一致性 |
| **C2_2** | dorm2_fast | +GI-KF | ❌ | ✅ | ❌ | GI-KF 跨场景泛化 |
| **C2_3** | dorm2_fast | +Pyramid | ❌ | ❌ | ✅ | Pyramid 跨场景泛化 |
| **C2_4** | dorm2_fast | +KF+Pyramid | ❌ | ✅ | ✅ | 组合策略跨场景 |
| **C2_5** | dorm2_fast | +ALL | ✅ | ✅ | ✅ | 全策略跨场景 |
| **C3_0** | hotel_fast1 | Baseline | ❌ | ❌ | ❌ | 酒店场景基线（纹理中等，挑战不同） |
| **C3_1** | hotel_fast1 | +IMU | ✅ | ❌ | ❌ | IMU 在中等纹理场景的效果 |
| **C3_2** | hotel_fast1 | +GI-KF | ❌ | ✅ | ❌ | GI-KF 在酒店布局下的表现 |
| **C3_3** | hotel_fast1 | +Pyramid | ❌ | ❌ | ✅ | Pyramid 在中等纹理的提升 |
| **C3_4** | hotel_fast1 | +KF+Pyramid | ❌ | ✅ | ✅ | 组合策略酒店场景 |
| **C3_5** | hotel_fast1 | +ALL | ✅ | ✅ | ✅ | 全策略酒店场景 |

**总计：44 个实验**

---

## 3. 快速开始

### 3.1 预览实验计划

```bash
python scripts/run_ablation.py --dry-run
```

### 3.2 先跑一个基线验证

```bash
# A组基线（TUM fr1/desk，最快跑完）
python scripts/run_ablation.py --experiment A1_0
```

跑完检查输出：

```bash
ls output/ablation/A1_0/*/
cat output/ablation/A1_0/*/ate_aligned.json
```

### 3.3 按场景跑

```bash
# 跑 A1 的全部 4 个策略
python scripts/run_ablation.py --experiment A1_0
python scripts/run_ablation.py --experiment A1_1
python scripts/run_ablation.py --experiment A1_2
python scripts/run_ablation.py --experiment A1_3

# 跑 B1 的全部 6 个策略
python scripts/run_ablation.py --experiment B1_0
python scripts/run_ablation.py --experiment B1_1
# ... B1_2 ~ B1_5
```

### 3.4 批量跑

```bash
# 跑全部 44 个实验（估计 8-12 小时）
python scripts/run_ablation.py

# 只跑 A 组（20 个，~5 小时）
python scripts/run_ablation.py --group A

# 中断后继续（自动跳过已完成）
python scripts/run_ablation.py
```

### 3.5 汇总结果

```bash
# Markdown 表格
python scripts/aggregate_results.py --format markdown

# 终端快速查看
python scripts/aggregate_results.py --format terminal
```

---

## 4. 实验策略解读

### 4.1 A 组（TUM，无 IMU）— 回答 GI-KF 和 Pyramid 各自的贡献

| 对比 | 计算 | 回答的问题 |
|------|------|-----------|
| A1_1 vs A1_0 | +GI-KF - Baseline | GI-KF 单独有用吗？ |
| A1_2 vs A1_0 | +Pyramid - Baseline | Pyramid 单独有用吗？ |
| A1_3 vs A1_0 | +Both - Baseline | 两者叠加效果？ |
| A1_3 vs A1_1+A1_2 | 实际 vs 单独贡献和 | 叠加还是互斥？ |

**A2-A5 重复上述模式**，验证跨场景泛化。

### 4.2 B 组（AzureKinect，有 IMU）— 回答 IMU 的贡献

| 对比 | 计算 | 回答的问题 |
|------|------|-----------|
| B1_1 vs B1_0 | +IMU - Baseline | IMU 单独有用吗？ |
| B1_2 vs B1_0 | +GI-KF - Baseline | GI-KF 单独有用吗？ |
| B1_3 vs B1_0 | +Pyramid - Baseline | Pyramid 单独有用吗？ |
| B1_4 vs B1_0 | +KF+Pyr - Baseline | 非 IMU 策略的组合贡献 |
| B1_5 vs B1_4 | +ALL - +KF+Pyr | **在 KF+Pyr 基础上加 IMU 的增量** |
| B1_5 vs B1_1 | +ALL - +IMU | 在 IMU 基础上加 KF+Pyr 的增量 |

关键对比是 **B1_5 vs B1_4**——这直接回答了 IMU 在已有 KF+Pyramid 的基础上是否还有额外收益。

### 4.3 C 组（FM，有 IMU，无 GT）— 独立验证 + IMU 泛化 + 快速运动

C 组与 B 组结构完全一致（6 策略），但 FM 数据集有两个关键不同：

1. **无真值位姿** → 无法计算 ATE，评估仅基于渲染指标
2. **快速运动** → 所有场景选择 fast 速度，帧间位移大

**为什么 FM 数据集对论文重要：**

| 角度 | 说明 |
|------|------|
| **IMU 泛化验证** | B 组是自采集 AzureKinect，C 组是公开 FM 数据集。在完全不同采集条件下验证 IMU 策略是否仍然有效，排除"只在自家数据上有效"的质疑 |
| **快速运动压力测试** | 帧间位移大时纯视觉跟踪容易丢失，IMU 的物理先验在这里最值钱。C1_1 vs C1_0 的 PSNR 差距应该比慢速场景更大 |
| **GI-KF 运动模糊过滤** | 快速运动更容易产生模糊帧，C1_2 (+GI-KF) 的关键帧数量应明显少于 C1_0 (Baseline 每帧都做)，且不损失渲染质量 |
| **跨场景一致性** | C1/C2/C3 三个不同室内布局，验证策略提升是否稳定（而非只在某个特定房间生效） |

**C 组核心对比（每个场景内）：**

| 对比 | 计算 | 回答的问题 | 预期结果 |
|------|------|-----------|---------|
| C1_1 vs C1_0 | +IMU - Baseline | **IMU 在快速运动下有用吗？** | PSNR ↑ 明显，因为视觉跟踪在快速运动下容易漂移 |
| C1_2 vs C1_0 | +GI-KF - Baseline | GI-KF 在富纹理场景的关键帧效率 | 关键帧数 ↓ 70-90%，PSNR ≈ |
| C1_3 vs C1_0 | +Pyramid - Baseline | Pyramid 在快速运动下的渲染质量 | PSNR ↑ 2-5% |
| C1_4 vs C1_0 | +KF+Pyr - Baseline | 两非 IMU 策略的组合贡献 | PSNR ↑，关键帧 ↓ |
| C1_5 vs C1_4 | +ALL - +KF+Pyr | **IMU 在已有 KF+Pyr 上的增量** | PSNR ↑ （核心对比） |
| C1_5 vs C1_1 | +ALL - +IMU | KF+Pyr 在已有 IMU 上的增量 | PSNR ↑ |
| C1_5 vs B1_5 | C 全开 vs B 全开 | **跨数据集一致性** | 趋势一致即可，绝对值可能不同 |

**跨场景验证（C1 vs C2 vs C3）：**

对每个策略编号 `_i`，比较 C1_i / C2_i / C3_i 的 PSNR 相对提升幅度。如果三个场景的趋势一致（比如 +IMU 在三个场景都提升 PSNR），说明策略泛化性好。如果某个场景反常（比如 hotel 场景 +IMU 下降），需要分析原因（纹理不足导致深度噪声被 IMU 放大？）。

---

## 5. 评估指标

| 指标 | 来源文件 | 方向 | 含义 |
|------|---------|------|------|
| ATE RMSE | `ate_aligned.json` | ↓ cm | 轨迹精度（A/B 组有，C 组无） |
| PSNR | `rendering_metrics.json` | ↑ dB | 渲染峰值信噪比 |
| SSIM | `rendering_metrics.json` | ↑ [0,1] | 结构相似性 |
| LPIPS | `rendering_metrics.json` | ↓ | 感知损失 |
| Depth L1 | `rendering_metrics.json` | ↓ m | 深度误差 |

---

## 7. 常见问题

### 后台运行

```bash
nohup python scripts/run_ablation.py > logs/ablation.log 2>&1 &
tail -f logs/ablation.log
```

### 单场景快速测试

在场景 YAML 中临时加 `frame_limit: 100` 加速验证：

```bash
# 创建测试配置
echo "frame_limit: 100" >> /tmp/test.yaml
# 或用命令行传（不支持 frame_limit 命令行参数，需改 YAML）
```

### 添加新场景

编辑 `scripts/run_ablation.py` 中的 `SCENES_A` / `SCENES_B` / `SCENES_C` 列表。
