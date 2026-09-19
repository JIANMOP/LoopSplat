## 4 实验设置

### 4.1 数据集与场景分组

本文在三个用途不同的室内 RGB-D 数据集上评测系统。TUM RGB-D 包含真实传感器序列和轨迹真值 [@sturm2012_tumrgbd]。实验选用 fr1/desk、fr1/desk2、fr1/room、fr2/xyz 和 fr3/long_office_household 五个序列，记为 A1–A5，共处理 8,454 帧。Replica 提供合成室内图像和精确相机轨迹 [@straub2019_replica]。实验采用常用的全部八个场景，即 office0–office4 与 room0–room2（R1–R8），共处理 16,000 帧。FMDataset 提供手持室内运动下采集的 RGB-D 与同步惯性测量 [@zhu2019_fmdataset]。其中 dorm1_fast1、dorm2_fast 和 hotel_fast1 构成 C1–C3，共处理 2,340 帧。

三组数据对应不同问题。TUM 用于考察改进在具有轨迹真值的真实 RGB-D 序列上的表现。Replica 用于检验预算匹配的关键帧延后策略在稳定合成场景上是否仍有价值，其基线本身已按五帧间隔建图。FMDataset 是正式实验中唯一含 IMU 输入的数据组，因此用于惯性消融。本实验所用的 C 组数据不含轨迹真值，故不报告 ATE 和 RPE。自采 Azure Kinect 材料仅用于工程定性检查，不纳入定量消融。

### 4.2 策略矩阵

Baseline 表示关闭三项改进的既有 LoopSplat 配置。TUM 与 Replica 比较 Baseline、GI-KF、Pyramid 和 GI-KF+Pyramid。FMDataset 另外加入 IMU 与 ALL，其中 ALL 同时启用 IMU、GI-KF 和 Pyramid。表 1 给出各策略开关。

**表 1 消融策略矩阵。勾选表示启用相应模块。**

| 数据组 | 策略 | IMU 旋转 | GI-KF | Pyramid |
|---|---|---:|---:|---:|
| TUM、Replica、FM | Baseline |  |  |  |
| FM | IMU | ✓ |  |  |
| TUM、Replica、FM | GI-KF |  | ✓ |  |
| TUM、Replica、FM | Pyramid |  |  | ✓ |
| TUM、Replica、FM | GI-KF+Pyramid |  | ✓ | ✓ |
| FM | ALL | ✓ | ✓ | ✓ |

除匹配各数据集基线节奏所需的间隔外，GI-KF 参数保持一致。TUM 与 FMDataset 的最小、稳定和最大间隔分别为 2、3、10 帧。Replica 的基线每五帧建图一次，因此采用 5、6、10 帧，并关闭跟踪支撑更新。信息分数权重、运动阈值、支撑更新长度和金字塔调度均按第 3 节设置。惯性策略的旋转权重为 0.001，平移权重为零。

### 4.3 实验协议与实现

表中每项对应一个场景-策略结果，“宏平均”表示对场景做无权算术平均。当前报告协议不估计运行间方差，也不支持显著性检验。所有策略使用相同的数据帧、评测视角列表、基础参数和 LoopSplat 既有功能，只改变表 1 中的策略开关。

实验使用云端提供的 22-GiB NVIDIA GeForce RTX 2080 Ti 实例与 16-GiB RTX 4080 SUPER。对于同一场景，各策略使用相同的 GPU 环境。TUM 的 A1、A2、A3、A5 及全部 FMDataset 实验使用 22-GiB RTX 2080 Ti 环境；TUM A4 与全部 Replica 实验使用 RTX 4080 SUPER。因此，时间和显存变化只在同一场景内解释；不同 GPU 场景之间的绝对用时仅作为运行背景。

实验程序保存实际生效配置、策略摘要、评测指标、耗时、峰值 PyTorch 已分配显存和持久关键帧判定。只有上述输出齐全且指标均为有限值的运行才进入论文结果。面向论文的提取程序为 70 个场景—策略配置各选择一条完整记录，其中 TUM 20 条、Replica 32 条、FMDataset 18 条。随后由统一 CSV 自动生成表格，使正文、表格和图中的数字来自同一数据源。

### 4.4 评测指标

TUM 与 Replica 的轨迹指标包括经过无尺度 Horn 刚性 \(SE(3)\) 对齐后的均方根绝对轨迹误差 ATE（厘米）。相对平移误差 RPE-t（米）和相对旋转误差 RPE-r（度）是在相邻帧对上计算的 RMSE，三者均为越低越好。FMDataset 因缺少参考轨迹而不报告这些指标。

渲染质量在固定的已观测帧列表上计算。PSNR 与 MS-SSIM 衡量像素和多尺度结构一致性，LPIPS 衡量感知距离 [@wang2003_msssim; @zhang2018_lpips]。PSNR、MS-SSIM 越高越好，LPIPS 越低越好。每次运行均使用自身估计位姿，并从未经全局细化的子图高斯直接拼接结果进行渲染。由于评测图像来自序列中已经观测过的帧，这些指标反映估计轨迹下的系统内部重投影一致性，不能单独代表世界坐标中的地图精度或未知视角泛化。Depth L1 是有效评测像素上的渲染深度平均绝对误差，越低越好。

效率指标包括端到端 SLAM 用时（小时）、由 torch.cuda.max_memory_allocated() 记录的峰值 PyTorch 已分配显存（GiB）、持久关键帧数量、子图数量，以及单独记录的跟踪支撑更新次数和用时。关键帧数量不包含支撑观测。主要效率结论中的百分比均以同场景 Baseline 为分母。

### 4.5 报告方式

表 2–4 先报告逐场景精度与质量，再给出跨场景无权宏平均；表 5 记录配对效率数据及每个场景所用 GPU。图 3 表示相对 Baseline 的指标变化，图 4 对照展示质量、关键帧和显存变化。数据集平均可能掩盖局部失败，因此结果讨论将明确保留 TUM A3 的退化和 FMDataset 上 IMU 的混合变化，而不会只依据总体均值。
