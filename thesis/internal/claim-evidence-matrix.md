# Claim–evidence matrix

This matrix is the wording contract for the manuscript masters. A later drafting task may use a claim only when its listed experiment or source is available at the listed location. Numeric conclusions require generated tables traced to the recorded run outputs.

| Field | Definition |
|---|---|
| `claim_id` | Stable identifier used to trace a manuscript statement to this contract. |
| `manuscript_section` | Intended master-manuscript location. |
| `supported_claim` | The bounded technical fact that evidence may support. |
| `experiment_or_source` | Evidence class or named source. |
| `exact_location` | Repository path and relevant section, symbol, or record. |
| `allowed_wording_en` / `allowed_wording_zh` | Approved English and Chinese formulations; later prose may be narrower but not broader. |
| `prohibited_extension` | A claim that the evidence does not license. |

| claim_id | manuscript_section | supported_claim | experiment_or_source | exact_location | allowed_wording_en | allowed_wording_zh | prohibited_extension |
|---|---|---|---|---|---|---|---|
| C01 | 3.1 Baseline and system overview | The work extends the inherited LoopSplat backend rather than replacing its baseline SLAM capabilities. | LoopSplat source and baseline paper | `README.md` system description; `paper/LoopSplat.md`; `thesis/2026-09-19-journal-paper-writing-design.md` §2.2 | “We build on the inherited LoopSplat submap, GS-GS registration, loop-closure, and pose-graph-optimization backend.” | “本文在既有 LoopSplat 的子图、GS-GS 配准、回环检测和位姿图优化后端上开展增强。” | Claiming a new SLAM paradigm or attributing inherited backend capabilities to the new strategies. |
| C02 | 3.2 Inertial-regularized RGB-D tracking | IMU contributes a preintegration-derived weak gyroscope rotation regularizer; translational IMU weight is zero. | FMDataset strategy configuration and experiment design | `scripts/run_ablation.py` `STRATEGIES_BC` (`_1`, `_5`); `md/ablation-guide.md` §1.2; design §2.2 and §5 | “The IMU term is a preintegration-derived weak gyroscope rotation regularizer with zero translational weight.” | “IMU 项为陀螺仪预积分导出的弱旋转正则，平移权重为零。” | “tightly coupled” visual–inertial state estimation; first use of IMU in 3DGS-SLAM; any claim of bias, velocity, gravity, or translation-state estimation. |
| C03 | 3.3 Visibility-aware persistent keyframe selection | GI-KF scores visibility, relative baseline, and motion state to select persistent keyframes under stated conditions. | Current implementation and GI-SLAM paper | `src/utils/keyframe_selection.py`; `src/utils/mapper_utils.py`; `paper/GI-SLAM.md`; design §2.2 and §5 | “The persistent-keyframe policy combines visibility, normalized baseline, and motion terms.” | “持久关键帧策略结合可见性、归一化基线和运动项。” | Claiming universal robustness, an unimplemented policy, or a guarantee for every scene. |
| C04 | 3.4 Lightweight tracking-support update | A non-persistent support observation can update the short-term map without entering the persistent keyframe, submap, or loop-closure records. | Experiment protocol and implementation | `md/ablation-guide.md` §1.2 and §5; `src/` keyframe/mapper implementation | “Tracking-support observations are short-term updates and are reported separately from persistent keyframes.” | “跟踪支撑观测是短期更新，与持久关键帧分开统计。” | Counting support observations as persistent keyframes or claiming that they participate in loop closure. |
| C05 | 3.6 Strategy combinations and dataset-specific budgets | Replica GI-KF uses a budget-matched schedule relative to the fixed-cadence baseline. | Replica strategy configuration | `scripts/run_ablation.py` `STRATEGIES_R`; design §2.3 | “On Replica, GI-KF is configured with a budget-matched persistent-keyframe schedule.” | “在 Replica 上，GI-KF 配置为与基线预算匹配的持久关键帧调度。” | Claiming an identical keyframe count before the experiment table verifies it. |
| C06 | 3.5 Image-pyramid-guided coarse-to-fine mapping | Mapping uses an RGB-D image pyramid and a use-count level schedule, not a hierarchical Gaussian representation. | Current implementation and design | `src/utils/mapper_utils.py` image/depth pyramid functions; `scripts/run_ablation.py` pyramid overrides; design §5 | “We use RGB-D image-pyramid supervision for coarse-to-fine mapping.” | “本文使用 RGB-D 图像金字塔监督进行从粗到细建图。” | A hierarchical Gaussian map, LOD map, or a claim that the map representation changes by pyramid level. |
| C07 | 4–5 TUM RGB-D and Replica | TUM and Replica can report trajectory metrics together with observed-view rendering, depth, and efficiency metrics when the recorded outputs pass audit. | Formal result outputs | `md/ablation-guide.md` §§5–7; design §3.3 | “For TUM RGB-D and Replica, we report ATE/RPE together with fixed-observed-view rendering, depth, and efficiency metrics.” | “对 TUM RGB-D 和 Replica，报告 ATE/RPE 及固定观测视角渲染、深度和效率指标。” | Calling the rendering evaluation novel-view synthesis; reporting values before audited result tables exist. |
| C08 | 5.3 FMDataset and inertial ablation | FMDataset has no trajectory ground truth and supports only observed-view image/depth quality and efficiency reporting in the formal ablation. | FMDataset protocol and design | `md/ablation-guide.md` §7.2; design §3.3 | “FMDataset results exclude trajectory metrics because no trajectory ground truth is available.” | “FMDataset 缺少轨迹真值，因此结果不包含轨迹指标。” | ATE/RPE, novel-view synthesis, or a trajectory-accuracy claim for FMDataset. |
| C09 | 5.6 Azure Kinect qualitative example | Azure Kinect material is for engineering usability and qualitative illustration only. | Design and experiment protocol | `md/ablation-guide.md` §1.4; design §§2.3, 3.3 | “Azure Kinect sequences are used only for qualitative engineering examples.” | “Azure Kinect 序列仅用于工程定性示例。” | Formal ablation, ATE/RPE, rigorous IMU validation, or a quantitative generalization claim. |

## Method equation trace

The English and Chinese method masters use the same equation numbers. The following rows record the implementation or canonical source that bounds each mathematical description.

| equation | manuscript content | implementation or literature trace | scope note |
|---|---|---|---|
| (1) | Gaussian parameters and covariance | src/entities/gaussian_model.py Gaussian parameter initialization and covariance accessors; src/utils/gaussian_model_utils.py build_scaling_rotation; [@kerbl2023_3dgs] | Three scale components remain trainable. |
| (2) | Color/depth alpha compositing | src/utils/utils.py render_gaussian_model; [@kerbl2023_3dgs] | Describes the renderer shared by tracking and mapping. |
| (3) | RGB-D visual tracking loss | src/entities/tracker.py tracking-loss construction and valid masks | Dataset configuration supplies the color/depth balance and masking options. |
| (4) | Gyroscope midpoint preintegration | src/entities/imu_preintegration.py preintegrate_imu; [@forster2017_preintegration] | Bias is configured and fixed; this is not joint bias estimation. |
| (5) | Rotation disagreement on SO(3) | src/entities/tracker.py compute_imu_loss_terms; src/entities/imu_preintegration.py so3_log | Residual scale is 0.01 rad in the formal configuration. |
| (6) | Robust weak rotation penalty | src/entities/tracker.py _huber_loss and compute_imu_loss_terms; [@huber1964_robust] | Formal rotation weight is 0.001 and Huber threshold is 0.1 rad. |
| (7) | Combined tracking objective | src/entities/tracker.py tracking objective; scripts/run_ablation.py STRATEGIES_BC | Formal translational IMU weight is zero. |
| (8) | Visibility/baseline/motion information score | src/utils/keyframe_selection.py gi_slam_keyframe_decision; src/utils/mapper_utils.py visibility and velocity helpers; [@liu2025_gislam] | Formal GI-KF motion is visual because use_imu_gyro remains false. |
| (9) | Interval safeguards and support trigger | src/utils/keyframe_selection.py forced_keyframe_decision; src/entities/gaussian_slam.py evaluate_gi_keyframe and should_run_tracking_support; scripts/run_ablation.py strategy overrides | The equation is a compact summary of ordered code branches; support observations are not persistent records. |
| (10) | Per-keyframe image-pyramid schedule | src/entities/mapper.py pyramid state and level selection; src/utils/mapper_utils.py build_image_pyramid, build_depth_pyramid, and render-setting scaling | Two sublevels and eight uses per sublevel are fixed in formal ablations. |
| (11) | Mapping loss and scale consistency | src/entities/mapper.py map optimization; src/entities/losses.py scale regularization; [@wang2004_ssim] | The scale term is a soft penalty and does not change the Gaussian parameterization. |

## Quantitative prose audit

Each row covers the aligned English and Chinese sentence group in the listed section. Values must be updated from the named generated asset rather than edited independently.

| audit_id | section and bounded sentence group | exact numeric source |
|---|---|---|
| Q01 | §4.1 TUM scene list and 8,454 processed frames | thesis/tables/result-source.csv, group A frame_count summed once per Baseline scene; thesis/internal/experiment-audit.md scene mapping |
| Q02 | §4.1 Replica scene list and 16,000 processed frames | thesis/tables/result-source.csv, group R frame_count summed once per Baseline scene; thesis/internal/experiment-audit.md scene mapping |
| Q03 | §4.1 FMDataset scene list and 2,340 processed frames | thesis/tables/result-source.csv, group C frame_count summed once per Baseline scene; thesis/internal/experiment-audit.md scene mapping |
| Q04 | §4.2 GI-KF interval values, score/pyramid settings, and IMU weights | scripts/run_ablation.py STRATEGIES_A, STRATEGIES_R, and STRATEGIES_BC |
| Q05 | §4.3 70 total configurations and 20/32/18 group split | thesis/internal/experiment-audit.md “Verified extraction snapshot”; thesis/tables/result-source.csv |
| Q06 | §5.1 TUM GI-KF+Pyramid macro keyframe, quality, and trajectory changes | thesis/tables/dataset-summary.csv rows A/Baseline and A/GI-KF+Pyramid; rounded display in thesis/tables/tum-results.md |
| Q07 | §5.1 A2 and A4 GI-KF+Pyramid comparisons | thesis/tables/tum-results.md rows A2/Baseline, A2/GI-KF+Pyramid, A4/Baseline, and A4/GI-KF+Pyramid |
| Q08 | §5.1 A3 Baseline, GI-KF, Pyramid, and combination failure values | thesis/tables/tum-results.md A3 rows |
| Q09 | §5.1 GI-KF TUM macro rendering and ATE values | thesis/tables/dataset-summary.csv rows A/Baseline and A/GI-KF; rounded display in thesis/tables/tum-results.md |
| Q10 | §5.2 Replica combined macro keyframe, trajectory, and quality changes | thesis/tables/dataset-summary.csv rows R/Baseline and R/GI-KF+Pyramid; rounded display in thesis/tables/replica-results.md |
| Q11 | §5.2 Replica per-scene counts and R5 changes | thesis/tables/replica-results.md, all Baseline and GI-KF+Pyramid rows, especially R5 |
| Q12 | §5.2 Replica Pyramid and GI-KF macro comparisons | thesis/tables/dataset-summary.csv R/Pyramid and R/GI-KF; rounded display in thesis/tables/replica-results.md |
| Q13 | §5.3 FMDataset ALL macro quality and keyframe changes | thesis/tables/dataset-summary.csv rows C/Baseline and C/ALL; rounded display in thesis/tables/fm-results.md |
| Q14 | §5.3 FMDataset C1 and C3 per-scene comparisons | thesis/tables/fm-results.md C1 and C3 rows |
| Q15 | §5.3 IMU-only macro and scene-level mixed response | thesis/tables/dataset-summary.csv C/Baseline and C/IMU; thesis/tables/fm-results.md Baseline and IMU rows |
| Q16 | §5.3 FMDataset GI-KF, Pyramid, GI-KF+Pyramid, and ALL comparisons | thesis/tables/dataset-summary.csv corresponding C rows; rounded display in thesis/tables/fm-results.md |
| Q17 | §5.4 TUM paired keyframe, time, and peak-memory reduction ranges | thesis/tables/efficiency-results.md A1–A5 Baseline and GI-KF+Pyramid rows |
| Q18 | §5.4 Replica paired ranges and macro memory | thesis/tables/efficiency-results.md R1–R8 Baseline and GI-KF+Pyramid rows; thesis/tables/dataset-summary.csv R rows |
| Q19 | §5.4 FMDataset ALL time/memory/keyframe values and IMU overhead | thesis/tables/efficiency-results.md C1–C3 Baseline, IMU, and ALL rows; thesis/tables/dataset-summary.csv corresponding C rows |
| Q20 | §5.4 dataset-level submap counts | thesis/tables/dataset-summary.csv submap_count for R and A strategies |
| Q21 | §5.6 70 controlled configurations | thesis/internal/experiment-audit.md “Verified extraction snapshot”; thesis/tables/result-source.csv |

## Global wording warnings

The manuscript must not claim a complete tightly coupled visual–inertial estimator, state-of-the-art performance, the first IMU use in 3DGS-SLAM, an isotropic Gaussian map, real-time mobile operation, or a guarantee across all scenes. These strings may occur here only as warnings or in the `prohibited_extension` field.
