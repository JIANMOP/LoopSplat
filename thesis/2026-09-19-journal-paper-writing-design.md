# LoopSplat 增强策略期刊论文写作设计

## 1. 写作目标

本项目将生成一篇不绑定具体期刊模板的完整科研论文母稿，主题为 LoopSplat 子图式 RGB-D 3DGS-SLAM 中三类增强策略的集成与实验分析。交付物包含内容一致的英文稿和中文稿、参考文献库、实验表格、论文插图及内部证据记录。英文稿用于后续 SCI 期刊投稿，中文稿用于导师审阅、修改和毕业材料准备。

论文以当前代码和已完成实验为主要证据。开题报告只用于说明研究动机和课题演变，不沿用其中已经被实现或实验否定的技术主张。正文不得虚构实验、引用、创新点或统计结果。

## 2. 论文定位

### 2.1 推荐题目

英文题目：

> Efficient Submap-Based 3D Gaussian SLAM with Visibility-Aware Keyframing, Coarse-to-Fine Mapping, and Inertial Rotation Regularization

中文题目：

> 融合可见性感知关键帧、粗到细建图与惯性旋转约束的高效子图式三维高斯SLAM

### 2.2 研究边界

论文定位为 LoopSplat 的多策略增强与实验分析，不宣称提出全新的 SLAM 范式。LoopSplat 原有的 3D Gaussian 子图、GS-GS 配准、回环检测和位姿图优化属于基线能力。本文研究的内容包括：

1. 将预积分导出的弱惯性旋转约束接入 RGB-D 跟踪目标；
2. 将可见性、相对基线和运动状态用于持久关键帧选择；
3. 在适用的数据集上使用轻量跟踪支撑更新，缓解稀疏持久关键帧造成的地图更新不足；
4. 使用多分辨率 RGB-D 图像金字塔进行从粗到细的子图优化；
5. 评估上述策略的单独作用、组合效果及场景依赖性。

以下表述不得出现在题目、摘要、贡献或结论中：

- 首次将 IMU 引入 3DGS-SLAM；
- 完整紧耦合视觉惯性状态估计；
- 各向同性高斯地图；
- 层次化 Gaussian 或 LOD 地图；
- 在移动设备上实时运行；
- 全面达到最优或 SOTA；
- 对所有场景保证鲁棒性。

### 2.3 贡献表述

论文采用三项贡献：

1. 在带回环的子图式 3DGS-SLAM 中统一集成惯性旋转正则、可见性感知关键帧和图像金字塔建图，并明确各策略的作用位置和启用条件。
2. 采用“持久关键帧与轻量跟踪支撑”机制，区分进入子图和回环的长期地图观测与只用于短期地图更新的支撑观测；Replica 则采用与固定步长基线匹配的关键帧预算。
3. 在 TUM RGB-D、Replica 和 FMDataset 上报告轨迹、固定观测视角渲染、深度误差及效率指标，分析单模块、组合策略和退化场景。自采 Azure Kinect 数据只用于工程可用性和定性展示。

## 3. 证据来源与事实口径

### 3.1 项目材料

- 开题报告：`thesis/彭飞扬学术学位硕士学位论文开题报告.pdf`
- 基线论文语料：`paper/LoopSplat.md`
- 相关方法语料：`paper/GI-SLAM.md`、`paper/Photo-SLAM.md`、`paper/Splatam.md`
- 当前实现：`src/`、`configs/`、`scripts/run_ablation.py`
- 实验协议：`md/ablation-guide.md`
- 正式输出：服务器 `/root/autodl-fs/output/ablation`

SCI 写作语料优先取自 CVF、IEEE、ACM、Springer、arXiv 论文原文和作者项目页。论文语料只用于学习术语、章节组织和论证方式，不复制原句。

### 3.2 实验口径

正式实验分为三组：

| 组别 | 数据集 | 场景 | 策略 |
|---|---|---:|---:|
| A | TUM RGB-D | 5 | Baseline、GI-KF、Pyramid、GI-KF+Pyramid |
| R | Replica | 8 | Baseline、GI-KF、Pyramid、GI-KF+Pyramid |
| C | FMDataset | 3 | Baseline、IMU、GI-KF、Pyramid、GI-KF+Pyramid、ALL |

当前结果来自每个配置的相同固定随机种子。论文可以报告逐场景结果和跨场景宏平均，但不得把跨场景平均称为多随机种子平均，也不得给出不存在的标准差。实验设置中说明所有对比采用相同固定随机种子，以减少随机性对消融比较的干扰。

源码提交哈希和源码指纹不进入论文正文、表格或补充材料。它们仅在内部证据记录中保留，用于结果追溯。正文只陈述可验证的公平性条件，例如相同数据、评估帧、基础参数和策略开关。

### 3.3 指标边界

- TUM 和 Replica 报告 ATE、RPE、PSNR、SSIM、LPIPS、Depth L1、关键帧数、子图数、运行时间和峰值显存；
- FMDataset 没有轨迹真值，只报告固定观测视角图像/深度质量与效率；
- Azure 没有可靠轨迹真值，不进入正式消融；
- 当前渲染指标来自固定观测视角，不称为 novel-view synthesis；
- 当前地图为未额外全局精修的子图高斯拼接，不虚构 mesh reconstruction 指标；
- 不使用“显著提升”，除非后续具备重复实验和统计检验。

## 4. 双语稿件结构

英文稿目标为 5,500 至 7,500 词，中文稿目标为 10,000 至 15,000 汉字。参考文献、公式、表格和图注不计入正文篇幅。

### Abstract / 摘要

- 英文 180 至 230 词，中文 300 至 450 字；
- 依次说明问题、方法、数据集、主要结果和适用边界；
- 只使用最终表格中可以复算的数值；
- 不使用“groundbreaking”“novel framework”“state of the art”等宣传性措辞。

### 1. Introduction / 引言

- 说明 3DGS-SLAM 在高保真地图、跟踪稳定性和在线计算预算之间的矛盾；
- 介绍 LoopSplat 的子图与回环能力及其固定映射预算；
- 引出惯性旋转约束、关键帧管理和粗到细建图三个研究问题；
- 给出准确、克制的贡献列表。

### 2. Related Work / 相关工作

按问题组织，而不是逐篇罗列：

1. 3D Gaussian Splatting 与 Gaussian SLAM；
2. 回环与全局一致的 3DGS-SLAM；
3. 视觉惯性 Gaussian SLAM；
4. 关键帧选择与地图更新；
5. 多尺度和粗到细优化。

### 3. Method / 方法

3.1 Baseline and system overview

3.2 Inertial-regularized RGB-D tracking

3.3 Visibility-aware persistent keyframe selection

3.4 Lightweight tracking-support update

3.5 Image-pyramid-guided coarse-to-fine mapping

3.6 Strategy combinations and dataset-specific budgets

### 4. Experimental Setup / 实验设置

- 数据集与场景；
- 策略矩阵；
- 实现参数；
- 硬件和运行环境；
- 固定随机种子与评估帧；
- 指标定义和方向；
- 公平比较条件。

### 5. Results and Discussion / 结果与讨论

5.1 TUM RGB-D results

5.2 Replica results

5.3 FMDataset and inertial ablation

5.4 Efficiency and keyframe budget

5.5 Component interaction and scene-dependent behavior

5.6 Azure Kinect qualitative example

### 6. Limitations / 局限性

说明弱旋转约束没有联合估计偏置、FMDataset 缺少轨迹真值、单固定随机种子不能估计方差、固定观测视角与新视角评测不同，以及策略在部分场景存在质量和效率权衡。

### 7. Conclusion / 结论

只总结实验支持的结论，不重复引言，不加入未经测试的未来性能承诺。

## 5. 公式设计

正文计划使用 9 至 11 组公式。所有符号先定义后使用，中英文版本保持相同编号。

1. 三维高斯参数与协方差分解；
2. 颜色和深度的 alpha compositing；
3. 掩膜后的 RGB-D 视觉跟踪损失；
4. 相机坐标系下的陀螺仪旋转预积分；
5. SO(3) 上的惯性旋转残差；
6. Huber 惯性正则与总跟踪目标；
7. 高斯视锥新颖度、深度归一化基线和运动惩罚构成的关键帧评分；
8. 首尾帧、最小间隔、稳定间隔、最大间隔和高速覆盖补偿的分段选择规则；
9. 轻量支撑更新的触发条件；
10. 图像金字塔分辨率与按使用次数升级的层级调度；
11. 由 L1、SSIM、深度和尺度正则组成的分层建图损失。

公式必须对应当前实现。平移预积分可以作为实现背景简述，但正式损失中明确令平移权重为零。高斯金字塔写成 RGB-D 图像金字塔监督，不写成层次 Gaussian 表示。

## 6. 表格与图表

### 6.1 主文表格

1. 数据集、场景、传感器、轨迹真值和评测指标表；
2. A/R/C 三组消融配置表；
3. TUM 逐场景与宏平均结果表；
4. Replica 逐场景与宏平均结果表；
5. FMDataset 逐场景与宏平均结果表；
6. 关键帧、子图、时间和显存效率表；
7. 相对 Baseline 的百分比变化汇总表。

表格使用程序从 JSON/YAML 输出生成，不手工重新录入数字。最优值和次优值只在比较方向明确时标注。

### 6.2 主文图形

1. 系统框图：RGB-D/IMU 输入、跟踪、GI-KF、支撑更新、金字塔建图、子图回环与输出；
2. 关键帧决策流程图；
3. 各策略相对 Baseline 的配对增量图；
4. 质量与效率 Pareto 图；
5. TUM 和 Replica 代表场景轨迹图；
6. Baseline 与组合策略的 RGB、深度和误差可视化；
7. Azure Kinect 定性重建图。

图形优先用 Python/Matplotlib 或矢量图生成，统一字体、颜色、尺寸和图例。若服务器结果已经包含轨迹图和映射可视化，则只读复制到论文附件，不重新启动实验。无 GPU 时不承诺重新渲染或导出新的全局 3DGS PLY。

## 7. 参考文献策略

参考文献目标为 25 至 35 篇，其中大部分为近五年的英文论文。所有条目必须能在 DOI、CVF、IEEE、ACM、Springer、arXiv 或期刊官网核验。

核心文献包括：

- 3D Gaussian Splatting；
- SplaTAM、GS-SLAM、Gaussian Splatting SLAM、Photo-SLAM；
- LoopSplat；
- ORB-SLAM3；
- MM3DGS SLAM、VINGS-Mono、GI-SLAM；
- Gaussian/Laplacian pyramid 的经典论文；
- 与回环、关键帧管理、RGB-D SLAM 和评测指标直接相关的正式论文。

GI-SLAM 标为 arXiv 预印本，不写成已正式发表。无法从可靠来源核验的开题报告参考文献不进入新稿。正文中的每个引用键都必须存在于 `references.bib`，每个 BibTeX 条目也必须在正文中实际使用或明确标记为备选。

## 8. Humanizer 与双语一致性

英文稿先根据证据表撰写，再生成语义对应的中文稿。中文稿不是逐句机械翻译，可以调整句序和术语解释，但数字、公式、引用和结论必须一致。

每章完成后执行 Humanizer 文件模式：

1. 标记 not-X-but-Y、模板式开头、强行三段式、夸张意义、宣传性词汇和聊天残留；
2. 保留所有有来源的事实、数值、公式和引用；
3. 调整句长、连接方式和段落节奏；
4. 检查润色前后是否丢失或新增事实；
5. 搜索残留的破折号、空泛结语、重复标题和装饰性粗体。

Humanizer 只能改善语体，不能用于规避学术诚信审查，也不能把缺失实验或不真实引用写得更像真人。

## 9. 文件结构

计划在 `thesis/` 中创建：

```text
thesis/
├── manuscript-en.md
├── manuscript-zh.md
├── references.bib
├── README.md
├── tables/
│   ├── result-source.csv
│   ├── tum-results.md
│   ├── replica-results.md
│   ├── fm-results.md
│   └── efficiency-results.md
├── figures/
│   ├── system-overview.pdf
│   ├── keyframe-flow.pdf
│   ├── metric-deltas.pdf
│   ├── quality-efficiency.pdf
│   └── qualitative/
└── internal/
    ├── experiment-audit.md
    ├── reference-audit.md
    └── claim-evidence-matrix.md
```

Markdown 是母稿格式，方便审阅和版本管理。确定目标期刊后，再从母稿转换到其 LaTeX 或 Word 模板。`internal/` 不属于投稿材料。

## 10. 写作与验证流程

1. 从服务器结果生成只读结果清单和表格源数据；
2. 建立参考文献库并逐条核验；
3. 建立“论文主张、实验数据、代码位置、参考文献”对应矩阵；
4. 完成方法公式和系统图；
5. 撰写英文方法、实验设置和结果；
6. 撰写英文引言、相关工作、摘要、局限性和结论；
7. 生成语义对应的中文稿；
8. 对中英文稿分别执行 Humanizer 编辑；
9. 自动检查数字、公式编号、表格引用、图引用和参考文献键；
10. 人工复核退化结果、百分比变化和贡献边界。

最终验收条件：

- 两版稿件章节、公式、图表、数字和引用对应；
- 所有论文数值可追溯到实验输出；
- 所有参考文献真实且书目信息完整；
- 不存在伪造的多随机种子均值或标准差；
- 不存在当前实现不支持的紧耦合、各向同性、实时或 SOTA 声明；
- Humanizer 不改变技术事实；
- 稿件中不含 TODO、TBD、占位引用或未解释缩写。
