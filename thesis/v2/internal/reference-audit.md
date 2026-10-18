# OARS-GS v2 正式文献审计

审计日期：2026-10-18。准入规则：推荐引用集合只含已正式发表且具有正式 DOI 的论文；题名、作者、会刊与年份以 DOI 注册记录和出版商页面交叉核对。`ACCEPTED` 仅表示出版状态和 DOI 已核验，不表示论文中的实验结论被本项目复现。正式 DOI 链接均可写成 `https://doi.org/<DOI>`；下表另保留出版商官方页面。

## 3D Gaussian splatting 与 Gaussian SLAM

| citation_key | formal_title | authors | venue | year | DOI | official_url | publication_status | intended_claim |
|---|---|---|---|---:|---|---|---|---|
| `kerbl2023_3dgs` | 3D Gaussian Splatting for Real-Time Radiance Field Rendering | Bernhard Kerbl; Georgios Kopanas; Thomas Leimkühler; George Drettakis | ACM Transactions on Graphics 42(4), Article 139 | 2023 | `10.1145/3592433` | https://dl.acm.org/doi/10.1145/3592433 | ACCEPTED — formal journal article | 显式各向异性 3-D Gaussian、可微 splatting、opacity compositing 与实时渲染基础；不是 SLAM 证据。 |
| `zhu2025_loopsplat` | LoopSplat: Loop Closure by Registering 3D Gaussian Splats | Liyuan Zhu; Yue Li; Erik Sandström; Shengyu Huang; Konrad Schindler; Iro Armeni | 2025 International Conference on 3D Vision (3DV), 156–167 | 2025 | `10.1109/3DV66043.2025.00020` | https://ieeexplore.ieee.org/document/11125565/ | ACCEPTED — formal conference paper | OARS-GS 继承的 Gaussian 子图注册回环与 pose-graph 全局校正；不得作为新贡献。 |
| `keetha2024_splatam` | SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM | Nikhil Keetha; Jay Karhade; Krishna Murthy Jatavallabhula; Gengshan Yang; Sebastian Scherer; Deva Ramanan; Jonathon Luiten | CVPR, 21357–21366 | 2024 | `10.1109/CVPR52733.2024.02018` | https://ieeexplore.ieee.org/document/10656349/ | ACCEPTED — formal conference paper | 在线 Gaussian RGB-D tracking/mapping 与重叠关键帧先例；不支持 LoopSplat 回环归属。 |
| `yan2024_gsslam` | GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting | Chi Yan; Delin Qu; Dan Xu; Bin Zhao; Zhigang Wang; Dong Wang; Xuelong Li | CVPR, 19595–19604 | 2024 | `10.1109/CVPR52733.2024.01853` | https://ieeexplore.ieee.org/document/10657581/ | ACCEPTED — formal conference paper | Gaussian RGB-D SLAM 与 Gaussian-selection coarse-to-fine tracking 先例；机制不同于本项目的 RGB-D 监督分辨率调度。 |
| `matsuki2024_gaussiansplatting` | Gaussian Splatting SLAM | Hidenobu Matsuki; Riku Murai; Paul H. J. Kelly; Andrew J. Davison | CVPR, 18039–18048 | 2024 | `10.1109/CVPR52733.2024.01708` | https://ieeexplore.ieee.org/document/10657715/ | ACCEPTED — formal conference paper | 对 Gaussian map 的直接相机优化背景；不支持 OARS-GS 的惯性或回环模块。 |
| `huang2024_photoslam` | Photo-SLAM: Real-Time Simultaneous Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras | Huajian Huang; Longwei Li; Hui Cheng; Sai-Kit Yeung | CVPR, 21584–21593 | 2024 | `10.1109/CVPR52733.2024.02039` | https://ieeexplore.ieee.org/document/10657868/ | ACCEPTED — formal conference paper | Gaussian-pyramid 多层训练先例；仅作动机，不能声称本项目采用相同 hyper-primitives 或调度。 |
| `sun2024_mm3dgsslam` | MM3DGS SLAM: Multi-modal 3D Gaussian Splatting for SLAM Using Vision, Depth, and Inertial Measurements | Lisong C. Sun; Neel P. Bhatt; Jonathan C. Liu; Zhiwen Fan; Zhangyang Wang; Todd E. Humphreys; Ufuk Topcu | IROS, 10159–10166 | 2024 | `10.1109/IROS58592.2024.10802389` | https://ieeexplore.ieee.org/document/10802389/ | ACCEPTED — formal conference paper | 正式 Gaussian-inertial SLAM 对照；不能证明 OARS-GS 联合估计速度、偏置、重力或惯性平移。 |
| `wu2025_vingsmono` | VINGS-Mono: Visual-Inertial Gaussian Splatting Monocular SLAM in Large Scenes | Ke Wu; Zicheng Zhang; Muer Tie; Ziqing Ai; Zhongxue Gan; Wenchao Ding | IEEE Transactions on Robotics 41, 5912–5931 | 2025 | `10.1109/TRO.2025.3613536` | https://ieeexplore.ieee.org/document/11176433/ | ACCEPTED — formal journal article | 完整大场景视觉惯性 Gaussian SLAM 对照类；强调本项目只使用弱旋转正则。 |

## Dense/RGB-D SLAM、关键帧、子图与全局一致性

| citation_key | formal_title | authors | venue | year | DOI | official_url | publication_status | intended_claim |
|---|---|---|---|---:|---|---|---|---|
| `newcombe2011_kinectfusion` | KinectFusion: Real-Time Dense Surface Mapping and Tracking | Richard A. Newcombe; Andrew Fitzgibbon; Shahram Izadi; Otmar Hilliges; David Molyneaux; David Kim; Andrew J. Davison; Pushmeet Kohli; Jamie Shotton; Steve Hodges | ISMAR, 127–136 | 2011 | `10.1109/ISMAR.2011.6092378` | https://ieeexplore.ieee.org/document/6162880/ | ACCEPTED — formal conference paper | 实时 dense RGB-D tracking 与 volumetric fusion 的历史背景；不支持 Gaussian 或学习式地图。 |
| `whelan2016_elasticfusion` | ElasticFusion: Real-Time Dense SLAM and Light Source Estimation | Thomas Whelan; Renato F. Salas-Moreno; Ben Glocker; Andrew J. Davison; Stefan Leutenegger | The International Journal of Robotics Research 35(14), 1697–1716 | 2016 | `10.1177/0278364916669237` | https://journals.sagepub.com/doi/10.1177/0278364916669237 | ACCEPTED — formal journal article | Surfel RGB-D tracking/fusion 与局部/全局回环校正的历史对照。 |
| `murartal2017_orbslam2` | ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras | Raúl Mur-Artal; Juan D. Tardós | IEEE Transactions on Robotics 33(5), 1255–1262 | 2017 | `10.1109/TRO.2017.2705103` | https://ieeexplore.ieee.org/document/7946260/ | ACCEPTED — formal journal article | 经典稀疏关键帧与回环背景；不支持 photorealistic dense reconstruction。 |
| `sucar2021_imap` | iMAP: Implicit Mapping and Positioning in Real-Time | Edgar Sucar; Shikun Liu; Joseph Ortiz; Andrew J. Davison | ICCV, IEEE/Crossref 6209–6218 | 2021 | `10.1109/ICCV48922.2021.00617` | https://ieeexplore.ieee.org/document/9710431/ | ACCEPTED — formal conference paper | 神经隐式 RGB-D 在线联合 tracking/mapping 与活动关键帧集。 |
| `zhu2022_niceslam` | NICE-SLAM: Neural Implicit Scalable Encoding for SLAM | Zihan Zhu; Songyou Peng; Viktor Larsson; Weiwei Xu; Hujun Bao; Zhaopeng Cui; Martin R. Oswald; Marc Pollefeys | CVPR, IEEE/Crossref 12776–12786 | 2022 | `10.1109/CVPR52688.2022.01245` | https://ieeexplore.ieee.org/document/9878912/ | ACCEPTED — formal conference paper | 分层局部特征网格背景；不能描述 OARS-GS 图像监督 pyramid。 |
| `sandstrom2023_pointslam` | Point-SLAM: Dense Neural Point Cloud-Based SLAM | Erik Sandström; Yue Li; Luc Van Gool; Martin R. Oswald | ICCV, IEEE/Crossref 18387–18398 | 2023 | `10.1109/ICCV51070.2023.01690` | https://ieeexplore.ieee.org/document/10377819/ | ACCEPTED — formal conference paper | 输入自适应神经点图与 TUM/Replica 评测惯例。 |
| `yang2022_voxfusion` | Vox-Fusion: Dense Tracking and Mapping with Voxel-Based Neural Implicit Representation | Xingrui Yang; Hai Li; Hongjia Zhai; Yuhang Ming; Yuqian Liu; Guofeng Zhang | ISMAR, 499–507 | 2022 | `10.1109/ISMAR55827.2022.00066` | https://ieeexplore.ieee.org/document/9995035/ | ACCEPTED — formal conference paper | Voxel neural隐式 dense tracking/mapping 的表示与效率对照。 |
| `wang2023_coslam` | Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM | Hengyi Wang; Jingwen Wang; Lourdes Agapito | CVPR, 13293–13302 | 2023 | `10.1109/CVPR52729.2023.01277` | https://ieeexplore.ieee.org/document/10204198/ | ACCEPTED — formal conference paper | 混合编码和跨帧全局 bundle adjustment；对照小规模选定关键帧集。 |
| `johari2023_eslam` | ESLAM: Efficient Dense SLAM System Based on Hybrid Representation of Signed Distance Fields | Mohammad Mahdi Johari; Camilla Carta; François Fleuret | CVPR, 17408–17419 | 2023 | `10.1109/CVPR52729.2023.01670` | https://ieeexplore.ieee.org/document/10205103/ | ACCEPTED — formal conference paper | 高效混合 SDF 表示与多尺度紧凑特征；不支持 Gaussian primitive。 |
| `tang2023_mipsfusion` | MIPS-Fusion: Multi-Implicit-Submaps for Scalable and Robust Online Neural RGB-D Reconstruction | Yijie Tang; Jiazhao Zhang; Zhinan Yu; He Wang; Kai Xu | ACM Transactions on Graphics 42(6), Article 246 | 2023 | `10.1145/3618363` | https://dl.acm.org/doi/10.1145/3618363 | ACCEPTED — formal journal article | 局部隐式子图限制在线优化和保持全局一致性的相关背景；不是 LoopSplat 继承证据。 |
| `liso2024_loopyslam` | Loopy-SLAM: Dense Neural SLAM with Loop Closures | Lorenzo Liso; Erik Sandström; Vladimir Yugay; Luc Van Gool; Martin R. Oswald | CVPR, 20363–20373 | 2024 | `10.1109/CVPR52733.2024.01925` | https://ieeexplore.ieee.org/document/10655306/ | ACCEPTED — formal conference paper | Dense neural SLAM 显式回环背景，与 LoopSplat 的 Gaussian 注册区分。 |
| `zhang2023_goslam` | GO-SLAM: Global Optimization for Consistent 3D Instant Reconstruction | Youmin Zhang; Fabio Tosi; Stefano Mattoccia; Matteo Poggi | ICCV, IEEE/Crossref 3704–3714 | 2023 | `10.1109/ICCV51070.2023.00345` | https://ieeexplore.ieee.org/document/10378579/ | ACCEPTED — formal conference paper | 回环与在线全局优化的相关背景。 |

## 视觉惯性、预积分、关键帧优化与鲁棒损失

| citation_key | formal_title | authors | venue | year | DOI | official_url | publication_status | intended_claim |
|---|---|---|---|---:|---|---|---|---|
| `campos2021_orbslam3` | ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual–Inertial, and Multimap SLAM | Carlos Campos; Richard Elvira; Juan J. Gómez Rodríguez; José M. M. Montiel; Juan D. Tardós | IEEE Transactions on Robotics 37(6), 1874–1890 | 2021 | `10.1109/TRO.2021.3075644` | https://ieeexplore.ieee.org/document/9440682/ | ACCEPTED — formal journal article | 完整 feature-based visual-inertial/multimap 系统对照；不证明本项目实现完整惯性状态。 |
| `forster2017_preintegration` | On-Manifold Preintegration for Real-Time Visual–Inertial Odometry | Christian Forster; Luca Carlone; Frank Dellaert; Davide Scaramuzza | IEEE Transactions on Robotics 33(1), 1–21 | 2017 | `10.1109/TRO.2016.2597321` | https://ieeexplore.ieee.org/document/7557075/ | ACCEPTED — formal journal article | 流形 IMU 预积分数学背景；不能据此声称 OARS-GS 实现完整 bias-aware VIO factor。 |
| `leutenegger2015_keyframevio` | Keyframe-Based Visual–Inertial Odometry Using Nonlinear Optimization | Stefan Leutenegger; Simon Lynen; Michael Bosse; Roland Siegwart; Paul Furgale | The International Journal of Robotics Research 34(3), 314–334 | 2015 | `10.1177/0278364914554813` | https://journals.sagepub.com/doi/10.1177/0278364914554813 | ACCEPTED — formal journal article | 有界窗口关键帧优化与变间隔惯性约束背景。年份按 2015 正式卷期，不用 online-first 2014。 |
| `huber1964_robust` | Robust Estimation of a Location Parameter | Peter J. Huber | The Annals of Mathematical Statistics 35(1), 73–101 | 1964 | `10.1214/aoms/1177703732` | https://projecteuclid.org/euclid.aoms/1177703732 | ACCEPTED — formal journal article | Huber 鲁棒估计基础；不支持本项目阈值选择或经验收益。 |

## 多分辨率与层次表示

| citation_key | formal_title | authors | venue | year | DOI | official_url | publication_status | intended_claim |
|---|---|---|---|---:|---|---|---|---|
| `burt1981_fastfilter` | Fast Filter Transform for Image Processing | Peter J. Burt | Computer Graphics and Image Processing 16(1), 20–51 | 1981 | `10.1016/0146-664X(81)90092-7` | https://www.sciencedirect.com/science/article/pii/0146664X81900927 | ACCEPTED — formal journal article | 经典低通/滤波变换与多分辨率图像处理基础。 |
| `burt1983_pyramid` | The Laplacian Pyramid as a Compact Image Code | Peter J. Burt; Edward H. Adelson | IEEE Transactions on Communications 31(4), 532–540 | 1983 | `10.1109/TCOM.1983.1095851` | https://ieeexplore.ieee.org/document/1095851/ | ACCEPTED — formal journal article | Gaussian/Laplacian pyramid 基础；不证明 OARS-GS 调度有效。 |
| `kerbl2024_hierarchical` | A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets | Bernhard Kerbl; Andreas Meuleman; Georgios Kopanas; Michael Wimmer; Alexandre Lanvin; George Drettakis | ACM Transactions on Graphics 43(4), Article 62 | 2024 | `10.1145/3658160` | https://dl.acm.org/doi/10.1145/3658160 | ACCEPTED — formal journal article | 3DGS 表示层级/LOD 对照；用于明确 OARS-GS 只变监督分辨率、保留一个活动 Gaussian 集。 |

## 数据集、对齐与指标

| citation_key | formal_title | authors | venue | year | DOI | official_url | publication_status | intended_claim |
|---|---|---|---|---:|---|---|---|---|
| `sturm2012_tumrgbd` | A Benchmark for the Evaluation of RGB-D SLAM Systems | Jürgen Sturm; Nikolas Engelhard; Felix Endres; Wolfram Burgard; Daniel Cremers | IROS, 573–580 | 2012 | `10.1109/IROS.2012.6385773` | https://ieeexplore.ieee.org/document/6385773/ | ACCEPTED — formal conference paper | TUM RGB-D 与 ATE/RPE 术语；不证明当前实现或数值正确。 |
| `zhu2019_fmdataset` | Real-Time Indoor Scene Reconstruction with RGBD and Inertial Input | Zunjie Zhu; Feng Xu; Chenggang Yan; Xinhong Hao; Xiangyang Ji; Yongdong Zhang; Qionghai Dai | ICME, 7–12 | 2019 | `10.1109/ICME.2019.00010` | https://ieeexplore.ieee.org/document/8784898/ | ACCEPTED — formal conference paper | 本项目惯性 ablation 使用的 RGB-D/inertial 数据集来源；当前实验无可用轨迹真值仍须由 CSV 空字段证明。 |
| `horn1987_absoluteorientation` | Closed-Form Solution of Absolute Orientation Using Unit Quaternions | Berthold K. P. Horn | Journal of the Optical Society of America A 4(4), 629–642 | 1987 | `10.1364/JOSAA.4.000629` | https://opg.optica.org/josaa/abstract.cfm?uri=josaa-4-4-629 | ACCEPTED — formal journal article | aligned ATE 前的无尺度刚体 SE(3) 对齐；不得写成尺度校正 Sim(3)。 |
| `wang2003_msssim` | Multiscale Structural Similarity for Image Quality Assessment | Zhou Wang; Eero P. Simoncelli; Alan C. Bovik | Thirty-Seventh Asilomar Conference on Signals, Systems and Computers, Vol. 2, 1398–1402 | 2003 | `10.1109/ACSSC.2003.1292216` | https://ieeexplore.ieee.org/document/1292216/ | ACCEPTED — formal conference paper | 评测所用 MS-SSIM；结果仍是 fixed-observed-view。 |
| `wang2004_ssim` | Image Quality Assessment: From Error Visibility to Structural Similarity | Zhou Wang; Alan C. Bovik; Hamid R. Sheikh; Eero P. Simoncelli | IEEE Transactions on Image Processing 13(4), 600–612 | 2004 | `10.1109/TIP.2003.819861` | https://ieeexplore.ieee.org/document/1284395/ | ACCEPTED — formal journal article | Mapping objective 的单尺度 SSIM；不要与评测 MS-SSIM 混用。 |
| `zhang2018_lpips` | The Unreasonable Effectiveness of Deep Features as a Perceptual Metric | Richard Zhang; Phillip Isola; Alexei A. Efros; Eli Shechtman; Oliver Wang | CVPR, 586–595 | 2018 | `10.1109/CVPR.2018.00068` | https://ieeexplore.ieee.org/document/8578166/ | ACCEPTED — formal conference paper | LPIPS 指标来源，越低越好；观察视角 LPIPS 不支持 novel-view 泛化。 |

## 写作引用路由

1. 继承后端：用 `zhu2025_loopsplat` 支持子图注册回环与 PGO，用 `kerbl2023_3dgs` 支持 Gaussian 表示；新旧贡献必须分开。
2. 惯性范围：用 `forster2017_preintegration` 解释流形预积分背景；`campos2021_orbslam3`、`leutenegger2015_keyframevio`、`sun2024_mm3dgsslam`、`wu2025_vingsmono` 只作完整视觉惯性系统的对照。OARS-GS 只能写成弱旋转正则，正式平移损失权重为零。
3. 关键帧/观测管理：`murartal2017_orbslam2`、`sucar2021_imap`、`keetha2024_splatam`、`leutenegger2015_keyframevio`、`wang2023_coslam` 提供背景；临时非持久 support 的事实只由代码审计支持。
4. Coarse-to-fine：`burt1981_fastfilter`、`burt1983_pyramid` 支持经典图像金字塔；`huang2024_photoslam`、`yan2024_gsslam` 是不同机制的先例；`kerbl2024_hierarchical` 仅用于区分表示层级。
5. 数据与指标：TUM/ATE/RPE 用 `sturm2012_tumrgbd`，刚体对齐用 `horn1987_absoluteorientation`，FM 用 `zhu2019_fmdataset`，MS-SSIM/SSIM/LPIPS 分别用 `wang2003_msssim`、`wang2004_ssim`、`zhang2018_lpips`。

## 第二版 BibTeX 元数据决策

`thesis/v2/references.bib` 是待 Task 6 按英文正文首次引用顺序裁剪、重排的候选库，共 33 条，逐条对应上表 `ACCEPTED` 记录。每条均保留纯 DOI 值和出版商官方 URL，并标记 `keywords = {candidate}`；三条 `REJECTED` 预印本和无 DOI 的 DROID-SLAM 均未进入该库。

Task 3 复核 DOI 注册元数据，并沿用上表对核心论文及三个创新点直接支撑文献所做的出版商/正式论文集页面核验。冲突按以下规则落库：

- SplaTAM 的 CVF HTML 索引省略 `Splat` 后的逗号，但 CVF 正式论文 PDF 和 DOI 注册题名均为 “SplaTAM: Splat, Track & Map …”；BibTeX 采用带逗号的正式题名。Photo-SLAM 的 CVF PDF 将 `Real-time` 写成句式大小写，而 DOI/IEEE 正式记录为 `Real-Time`；BibTeX 采用 DOI/IEEE 形式。其余作者顺序和年份一致。
- 3DGS、MIPS-Fusion 和层次化 3DGS 的 DOI 注册页码分别显示 `1–14`、`1–16`、`1–15`，但 ACM 正式卷期按 Article 139、246、62 编号；BibTeX 使用 `articleno`，不把文章内页伪装成卷内连续页码。
- iMAP、NICE-SLAM、Point-SLAM、GO-SLAM 的 CVF OA 页面与 IEEE/Crossref 最终页码不一致；继续采用上表已核验的 IEEE/Crossref 页码 `6209–6218`、`12776–12786`、`18387–18398`、`3704–3714`。
- KinectFusion 的 DOI 注册作者串误写 `Pushmeet Kohi`，正式论文为 `Pushmeet Kohli`；TUM RGB-D 的 DOI 注册导出丢失 `Jürgen` 的变音符；Asilomar DOI 注册容器名含 `Thrity-Seventh` 拼写错误且发布日期字段缺失。BibTeX 分别采用正式论文/论文集中的 `Pushmeet Kohli`、`Jürgen Sturm`、`Thirty-Seventh …` 和 2003 年。
- Keyframe-Based Visual–Inertial Odometry 的 online-first 年份是 2014，但正式卷期为 2015；BibTeX 沿用正式卷期年份。作者姓名中的 `Sandström`、`Gómez Rodríguez`、`Tardós`、`François` 等变音符以正式出版显示为准，不沿用 DOI 导出中的 ASCII 降级形式。

Replica 原始报告没有核验到正式会刊/DOI。正文可写：使用八个 Replica 场景并遵循已发表 Gaussian/RGB-D SLAM 工作中的评测协议，精确场景、采帧和评测帧选择放入可复现材料；数据访问 URL 放在 data-availability statement。可引用 `keetha2024_splatam`、`yan2024_gsslam`、`matsuki2024_gaussiansplatting`、`sandstrom2023_pointslam` 支持“采用的评测协议”，但不能把它们写成 Replica 创建论文。

## REJECTED 审计记录：不得进入推荐引用集合

下列三项只有 arXiv 预印本记录。`10.48550/arXiv.*` 若由索引展示，仅标识预印本，不是正式出版 DOI。

| citation_key | formal_title | authors | venue | year | DOI | official_url | publication_status | intended_claim |
|---|---|---|---|---:|---|---|---|---|
| `yugay2023_gaussianslam` | Gaussian-SLAM: Photo-Realistic Dense SLAM with Gaussian Splatting | Vladimir Yugay; Yue Li; Theo Gevers; Martin R. Oswald | arXiv:2312.10070 | 2023 | 无正式出版 DOI | https://arxiv.org/abs/2312.10070 | REJECTED — arXiv-only；截至审计日未核验到正式后继版本 | 不得作为正式 Gaussian SLAM 证据；一般性背景改用已 ACCEPTED 的 SplaTAM、GS-SLAM、Gaussian Splatting SLAM、Photo-SLAM。 |
| `liu2025_gislam` | GI-SLAM: Gaussian-Inertial SLAM | Xulang Liu; Ning Tan | arXiv:2503.18275 | 2025 | 无正式出版 DOI | https://arxiv.org/abs/2503.18275 | REJECTED — arXiv-only；截至审计日未核验到正式后继版本 | 不得作为 OARS-GS 关键帧/惯性方法依据；正式 Gaussian-inertial 背景用 MM3DGS SLAM 与 VINGS-Mono，项目专属逻辑用代码。 |
| `straub2019_replica` | The Replica Dataset: A Digital Replica of Indoor Spaces | Julian Straub; Thomas Whelan; Lingni Ma; Yufan Chen; Erik Wijmans; Simon Green; Jakob J. Engel; Raúl Mur-Artal; Carl Ren; Shobhit Verma; Anton Clarkson; Mingfei Yan; Brian Budge; Yajie Yan; Xiaoqing Pan; June Yon; Yuyang Zou; Kimberly Leon; Nigel Carter; Jesús Briales; Tyler Gillingham; Elias Mueggler; Luis Pesqueira; Manolis Savva; Dhruv Batra; Hauke M. Strasdat; Renzo De Nardi; Michael Goesele; Steven Lovegrove; Richard Newcombe | arXiv:1906.05797 | 2019 | 无正式出版 DOI | https://arxiv.org/abs/1906.05797 | REJECTED — arXiv-only dataset report；DBLP 为 CoRR/informal，未核验到正式后继版本 | 数据 URL 进入 data-availability；正文的评测协议仅引用正式发表的相关 SLAM 论文。 |

`teed2021_droidslam` 是正式 NeurIPS 2021 论文，但本次没有核验到 proceedings DOI，因此不属于“正式 DOI 必需”的推荐集合；不得编造 DOI。CVF OA 与 IEEE/Crossref 页码不一致时，本审计采用 DOI/IEEE-final 页码：iMAP 6209–6218、NICE-SLAM 12776–12786、Point-SLAM 18387–18398、GO-SLAM 3704–3714。
