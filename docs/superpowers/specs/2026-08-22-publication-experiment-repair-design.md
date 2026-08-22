# LoopSplat 投稿级实验修复设计

日期：2026-08-22  
目标：将当前 LoopSplat + GI-KF + Gaussian Pyramid + IMU 实验修复为可支撑学校 2/4 分档投稿的可复现证据链。

## 1. 范围与成功标准

本次工作只修复会影响方法正确性、评测公平性和实验可追溯性的代码，不重构 LoopSplat 无关模块，也不增加新的论文功能。

完成标准：

1. FM 时间戳统一为秒，跟踪器使用相邻 RGB 帧区间内的全部 IMU 样本。
2. IMU 旋转残差保持 PyTorch 可微，速度和预积分状态每帧只提交一次。
3. 顶层 `gaussian_pyramid` 配置显式传入 Mapper；启用时确实执行低分辨率到全分辨率的渐进优化。
4. FM 的第二帧不再使用 dummy identity pose，而是从第二帧开始正常跟踪。
5. 无有效 GT 的数据集不生成或汇总 ATE；TUM 报告 ATE 与 RPE。
6. 不同策略使用相同、去重的帧集合计算 observed-view 渲染指标。
7. 消融运行可安全续跑，结果汇总能识别实际输出目录和全局渲染指标。
8. 每个正式实验可追溯到 git commit、配置、seed、命令、硬件、运行日志和运行状态。
9. 本机 `loop_splat` 环境的 CUDA 单元测试和小规模 GPU 冒烟通过；服务器 FM/TUM GPU 复核通过后才允许启动正式实验矩阵。

## 2. 非目标

- 不实现完整紧耦合 VIO、在线 bias 联合优化或滑动窗口 bundle adjustment。
- 不把 FM 的 observed-view 指标表述为 novel-view synthesis。
- 不为 FM 或缺少真实 GT 的 Azure 数据生成伪轨迹精度。
- 不在本阶段选择投稿期刊、撰写论文正文或美化论文图表。
- 不修改原始数据文件。

## 3. IMU 数据契约

### 3.1 统一接口

带 IMU 的数据集提供以下语义接口：

```python
get_imu_measurements(start_frame_id: int, end_frame_id: int) -> IMUInterval
```

`IMUInterval` 包含：

- `timestamps_s`：严格递增的秒级时间戳；
- `accelerations`：单位为 m/s^2，位于 IMU 坐标系；
- `angular_velocities`：单位为 rad/s，位于 IMU 坐标系；
- `dt_s`：图像帧间隔；
- `valid` 和 `reason`：是否可用于该帧融合及不可用原因。

数据集加载时完成时间单位归一化，不允许 Tracker 猜测时间单位。FM 原始微秒除以 `1e6`；Azure 时间戳单位由配置显式声明。

区间查询包含覆盖 `[t_start, t_end]` 的边界样本，在两个图像时间点线性插值边界测量，并使用中点积分。样本不足两个、时间倒序、`dt_s <= 0` 或帧间隔超过配置阈值时返回无效区间。

### 3.2 标定与噪声

FM 配置显式保存：

- 相机到 IMU 的刚体变换及其方向约定；
- 加速度计、陀螺仪 bias 初值；
- 测量噪声；
- 重力大小；
- 图像与 IMU 时间偏移，默认 0 秒。

内部统一使用 `T_cam_imu` 表示“将 IMU 坐标中的点变换到相机坐标”。FM 文件给出的其他方向矩阵在配置加载阶段只转换一次。平移外参用于 IMU pose 与 camera pose 的转换；角速度和比力只应用旋转部分。

## 4. IMU 先验与 Tracker 状态

### 4.1 帧间预积分

在相邻 RGB 帧之间，对全部 IMU 样本进行中值预积分，输出：

- `delta_R_imu`：IMU 帧间相对旋转；
- `delta_v_imu`：去 bias 后的速度增量；
- `delta_p_imu`：去 bias 后的位置增量；
- `total_dt`：实际积分时长。

预积分在固定 bias 下完成。世界系重力通过初始静止窗口的平均加速度估计方向；若静止检测不成立，则只启用旋转先验，平移先验保持关闭。这样避免在快速起步时用错误重力制造更大漂移。

### 4.2 可微残差

候选相机位姿与 IMU 预测位姿之间计算：

- 旋转残差：`Log(R_imu_pred^T R_opt)`；
- 平移残差：仅在重力和速度状态有效时启用；
- 两项均使用 Huber loss。旋转残差除以配置的 `rot_residual_scale_rad`，平移残差除以 `trans_residual_scale_m`，再分别乘 `lambda_imu_rot` 和 `lambda_imu_trans`；尺度参数必须为有限正数。

SO(3) 指数映射和对数映射使用纯 PyTorch 实现，不允许 `detach()`、NumPy 或 SciPy 出现在 loss 路径中。测试必须证明旋转参数获得有限且非零梯度。

### 4.3 状态生命周期

Tracker 将持久状态与 loss 前向完全分离：

1. 帧开始时读取上一帧已提交状态并构造一次 `IMUPrediction`；
2. tracking 的所有优化迭代只读该预测，不修改速度、bias 或时间戳；
3. 选择最终最佳位姿后调用一次 `commit_imu_state()`；
4. 无效 IMU 区间不更新惯性状态，并记录原因。

因此 tracking iteration 数不会改变物理积分结果。

### 4.4 鲁棒降权与日志

IMU 先验不是静默开关。每帧记录 `valid`、样本数、积分时长、旋转/平移残差和实际权重。

出现以下情况时该帧自动禁用 IMU：

- 时间区间无效；
- IMU 样本不足；
- 角速度或加速度包含非有限值；
- 预测运动超过配置的物理上限；
- 初始重力估计不可用时请求平移约束。

鲁棒 loss 处理测量离群点，但不得掩盖时间单位或配置错误；数据契约错误直接失败。

## 5. Gaussian Pyramid 配置与调度

当前代码存在确定的配置边界错误：`GaussianSLAM` 仅将 `config["mapping"]` 传给 `Mapper`，但 `Mapper` 在这个 mapping 子字典中读取顶层 `gaussian_pyramid`，导致所有实验中的 `_pyramid_enabled` 实际恒为 `False`。

修复后由 `GaussianSLAM` 将 mapping 配置和 pyramid 配置作为两个显式参数传给 `Mapper`，不移动现有 YAML 层级。Mapper 初始化时验证：

- `enabled` 必须是布尔值；
- `num_sub_levels >= 1`；
- `uses_per_level >= 1`；
- 启用时打印并写入 manifest 的 effective pyramid 配置。

渐进优化保持现有“1/4 分辨率、1/2 分辨率、全分辨率”语义，但增加以下正确性要求：

1. 每个新关键帧都建立与配置一致的图像、有效深度 mask 和 render settings 金字塔；
2. 深度下采样使用有效像素加权，零深度不得与有效深度双线性混合成伪几何；
3. 每个关键帧按 `uses_per_level` 从最粗层依次升级，计数耗尽后稳定使用全分辨率；
4. Baseline 禁用时不得构建金字塔或改变原始优化路径；
5. 每个 run 在 manifest 中保存请求开关和 Mapper 实际生效开关，二者不一致时立即失败。

现有所有标记为 `+Pyramid`、`+KF+Pyramid` 和 `+ALL` 的结果因开关未传入 Mapper，不作为正式消融证据，修复后全部重跑。

## 6. 无 GT 数据集初始化

数据集增加 `has_ground_truth` 属性。TUM 为真，FM 为假；Azure 只有成功加载并验证真实轨迹后才为真。

GaussianSLAM 初始化规则：

- 第 0 帧建立世界原点；
- `gt_camera=true` 且数据集有 GT 时可直接使用 GT；
- 其他情况从第 1 帧开始调用正常 Tracker，不再无条件采用数据集 pose；
- dummy identity 只作为接口占位，不能进入评测或第二帧初始化。

## 7. 公平评测协议

### 7.1 轨迹

Evaluator 在 `has_ground_truth=false` 时明确记录 `trajectory.status=skipped_no_ground_truth`，不写 `ate.json` 或 `ate_aligned.json`。

有 GT 时输出：

- ATE RMSE；
- translational RPE RMSE；
- rotational RPE RMSE；
- 有效 pose pair 数和对齐方式。

### 7.2 渲染

所有策略使用同一个由配置生成的评测帧列表：默认全序列按固定 stride 取样，frame id 去重并排序。该列表保存为结果文件，汇总时校验不同策略列表完全一致。

全局地图对这些固定 observed frames 渲染并计算 PSNR、MS-SSIM、LPIPS 和有效深度像素上的 Depth L1。指标名称明确使用 `observed_view`，不能标注为 NVS。

原有按 `submap_keyframes` 计算的指标可保留为诊断项，但不能进入主消融表。

## 8. 实验产物与汇总

每个 run 使用不可覆盖目录：

```text
output/ablation/<experiment_id>/seed_<seed>/<run_id>/
```

`run_id` 由 UTC 时间和短随机后缀组成。目录至少包含：

- `config.yaml`；
- `manifest.json`：git commit、dirty 状态、命令、seed、配置 hash、Python/CUDA/PyTorch/GPU；
- `run.log`；
- `status.json`：running/succeeded/failed、开始/结束时间、wall time、峰值 GPU 显存、错误摘要；
- 轨迹、地图和评测 JSON；
- `evaluation_frame_ids.json`。

续跑逻辑只跳过 `status=succeeded` 且必要指标齐全的 run。失败或不完整结果不得被视为完成，也不得自动覆盖。

聚合脚本直接发现上述 run 目录，同时兼容读取现有 flat legacy 结果；正式表只采用新协议结果。汇总校验 seed、配置 hash、评测帧集合、GSR 次数和代码 commit，检测到混杂即报错。

## 9. 测试策略

### 9.1 本机 GPU 测试

本机使用 `conda run -n loop_splat` 运行完整 pytest 套件。涉及 IMU 预积分、SO(3) 和 Tracker tensor 状态的测试强制使用 CUDA，若 CUDA 不可用则测试失败而不是跳过；结果发现、manifest 等纯文件逻辑在同一套件内运行，但不人为搬到显存。

新增最小测试集：

1. FM 微秒正确转换为秒，图像约 30 Hz、IMU 约 200 Hz；
2. 帧间区间查询包含正确样本且输出严格递增；
3. 常角速度的预积分旋转与解析值一致；
4. 静止 IMU 经重力补偿后位置漂移在容差内；
5. 重复计算 loss 不改变持久状态，commit 只发生一次；
6. SO(3) 旋转残差对优化参数具有有限非零梯度；
7. 顶层 Pyramid 开关能传到 Mapper，关闭时保持 Baseline 路径；
8. 金字塔各层的图像、深度 mask、render 尺寸和渐进计数符合配置；
9. 无 GT 数据跳过轨迹评测；
10. 固定评测 frame id 在不同关键帧策略下完全一致；
11. runner 正确识别成功、失败和 legacy 结果；
12. aggregator 读取全局 observed-view 指标并拒绝混杂配置。

单元测试通过后，在本机 RTX 4060 Laptop GPU 上执行 FM 和 TUM 各 5--10 帧的 Baseline 冒烟，以及 FM 同序列 IMU 冒烟。由于本机显存为 8 GB，该阶段只验证 CUDA 路径、有限 loss、输出完整性和 IMU 状态生命周期，不用其耗时或显存结果与服务器正式实验横向比较。

### 9.2 服务器 GPU 复核

在 `loop_splat` Conda 环境中依次运行：

1. 测试集；
2. FM 10--30 帧 Baseline 冒烟；
3. 同序列 IMU 冒烟；
4. 同序列 Pyramid 冒烟，核验日志和 manifest 显示实际生效；
5. TUM 10--30 帧 Baseline 冒烟；
6. 对比 IMU/Pyramid 开关下时间、显存、loss 有限性和输出完整性。

任何本机或服务器测试出现 NaN、无限 loss、伪 ATE、评测帧不一致或状态重复更新，都阻止正式实验启动。

## 10. 正式实验阶段门槛

代码修复完成不等于达到投稿标准。只有在服务器冒烟通过后才冻结 commit 和配置，随后运行：

- TUM 至少 3 个场景的 Baseline/GI-KF/Pyramid/KF+Pyramid；
- FM 至少 3 个场景的 Baseline/IMU/GI-KF/Pyramid/KF+Pyramid/ALL；
- Baseline 与最终方法至少 3 个 seed；
- 所有方法使用相同评测帧、GSR 次数、分辨率和硬件预算。

正式结果必须同时报告精度、失败率、关键帧数、子图数、总耗时和峰值显存。若修复后 IMU 仍在多个场景稳定退化，则停止将 IMU 作为正向贡献，转为失效分析或从最终方法中移除。

## 11. Git 与服务器发布流程

1. 本机在隔离分支中按测试驱动方式实现；
2. 本机完整测试通过后提交并推送 GitHub；
3. 服务器确认工作区干净，启用 GitHub 网络加速后拉取同一 commit；
4. 服务器验证 `git rev-parse HEAD` 与本机冻结 commit 一致；
5. 使用 `conda run -n loop_splat` 执行测试和冒烟；
6. 正式实验使用后台会话运行，结果持续写入独立 run 目录。

服务器不得在未提交代码上直接热修；任何必要修复均回到本机，重新走测试、提交、推送、拉取和 commit 一致性校验。
