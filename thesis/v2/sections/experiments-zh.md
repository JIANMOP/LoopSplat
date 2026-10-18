# 4. 实验

## 4.1. 实验设置

**数据集。** 我们评测三组室内 RGB-D 数据。TUM RGB-D 提供真实序列和轨迹真值 [@sturm2012_tumrgbd]。五个序列为 fr1/desk、fr1/desk2、fr1/room、fr2/xyz 和 fr3/long_office_household，记为 A1--A5。Replica 包含八个合成场景 office0--office4 与 room0--room2，记为 R1--R8。由于其数据集报告在本次审计中不是带 DOI 的已接受正式参考文献，我们引用已发表的高斯和 RGB-D SLAM 工作来说明所采用的评测惯例，而不将该报告作为正式文献 [@keetha2024_splatam; @yan2024_gsslam; @sandstrom2023_pointslam]。FMDataset 提供同步 RGB-D 和惯性测量 [@zhu2019_fmdataset]；dorm1_fast1、dorm2_fast 和 hotel_fast1 构成 C1--C3。实验记录不包含 FMDataset 轨迹真值，因而其 ATE 和 RPE 单元保持缺失，不以零代替。另行采集的 Azure Kinect 材料仅为工程说明，不属于消融实验。

**策略。** Baseline 是关闭三项 OARS-GS 决策的继承 LoopSplat 配置。TUM 和 Replica 比较 Baseline、GI-KF、Pyramid 及 GI-KF+Pyramid。FMDataset 还包括 IMU 和 ALL，其中 ALL 启用惯性旋转正则、GI-KF 和 Pyramid。

| 数据集组 | 策略 | 惯性旋转 | GI-KF | Pyramid |
|---|---|---:|---:|---:|
| TUM, Replica, FM | Baseline |  |  |  |
| FM | IMU | ✓ |  |  |
| TUM, Replica, FM | GI-KF |  | ✓ |  |
| TUM, Replica, FM | Pyramid |  |  | ✓ |
| TUM, Replica, FM | GI-KF+Pyramid |  | ✓ | ✓ |
| FM | ALL | ✓ | ✓ | ✓ |

Replica 相对其继承基线使用预算匹配的关键帧节奏，并禁用临时支撑路径。TUM 和 FMDataset 启用经审计的支撑策略。其余策略语义和形式化参数均遵循第 3 节；尤其是，IMU 设置的优化贡献仅为旋转，而金字塔改变 RGB-D 监督，不改变高斯表示。

**选择与聚合协议。** 规范化证据中，每个场景和策略均使用固定的已选种子。每条被选记录均使用 seed 0；TUM、Replica 与 FMDataset 分别有 20、32 和 18 条记录。策略宏值是场景值的无权算术平均，场景而非随机种子是平均单位。结果是对这一固定选择的描述性比较，不是随机种子平均、不确定性估计或显著性检验。所有百分比变化以同场景 Baseline 值为分母；若为指定数据集宏比较，则以相同场景集合上的 Baseline 宏值为分母。

**硬件与资源协议。** TUM A1、A2、A3、A5 以及所有 FMDataset 策略记录于 NVIDIA GeForce RTX 2080 Ti；TUM A4 和所有 Replica 策略记录于 NVIDIA GeForce RTX 4080 SUPER。每个同场景策略比较保持所记录的硬件一致。因此，我们以同场景配对变化解释资源结果，而不同 GPU 型号上的绝对值只作环境描述。运行时间是端到端 SLAM 耗时。峰值内存指 torch.cuda.max_memory_allocated() 返回的最大 PyTorch 分配量，而非整个设备占用。持久关键帧计数不包括临时跟踪支撑更新；后者的次数和耗时另行记录。

**指标。** TUM 与 Replica 的跟踪使用均方根绝对轨迹误差 (ATE)、相对平移误差 (RPE-t) 和相对旋转误差 (RPE-r)。ATE 在无尺度校正的刚性 (SE(3)) 对齐后报告 [@horn1987_absoluteorientation]，数值越低越好。因所需参考轨迹不在记录中，FMDataset 没有跟踪结果。

渲染在每种方法自身估计轨迹下的固定采样输入观测列表上进行。PSNR 和 MS-SSIM 衡量光度与多尺度结构一致性，LPIPS 衡量感知距离，Depth L1 衡量有效像素上的渲染深度误差 [@wang2003_msssim; @zhang2018_lpips]。PSNR 和 MS-SSIM 越高越好；LPIPS 和 Depth L1 越低越好。被评测图像是由直接、未经细化的子图拼接渲染出的已观测视角。这些测量不证明未观测视角泛化、独立全局地图几何或网格精度。

详细的逐场景结果见 [TUM 表](../../tables/tum-results.md)、[Replica 表](../../tables/replica-results.md)、[FMDataset 表](../../tables/fm-results.md) 和[配对资源表](../../tables/efficiency-results.md)。
