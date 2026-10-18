# 4. Experiments

## 4.1. Experimental setup

**Datasets.** We evaluate three indoor RGB-D groups. TUM RGB-D supplies real sequences and trajectory ground truth [@sturm2012_tumrgbd]. The five sequences are fr1/desk, fr1/desk2, fr1/room, fr2/xyz, and fr3/long_office_household, denoted A1--A5. Replica contributes eight synthetic scenes, office0--office4 and room0--room2, denoted R1--R8. Because its dataset report is not an accepted DOI-bearing formal reference in our audit, we cite published Gaussian and RGB-D SLAM work for the adopted evaluation convention rather than treating that report as formal literature [@keetha2024_splatam; @yan2024_gsslam; @sandstrom2023_pointslam]. FMDataset supplies synchronized RGB-D and inertial measurements [@zhu2019_fmdataset]; dorm1_fast1, dorm2_fast, and hotel_fast1 form C1--C3. The experiment records contain no FMDataset trajectory ground truth, and their ATE and RPE cells remain missing rather than being replaced by zeros. Separately captured Azure Kinect material is an engineering illustration only and is not part of the ablation.

**Strategies.** Baseline is the inherited LoopSplat configuration with the three OARS-GS decisions disabled. TUM and Replica compare Baseline, GI-KF, Pyramid, and GI-KF+Pyramid. FMDataset additionally includes IMU and ALL, where ALL enables inertial rotation regularization, GI-KF, and Pyramid.

| Dataset group | Strategy | Inertial rotation | GI-KF | Pyramid |
|---|---|---:|---:|---:|
| TUM, Replica, FM | Baseline |  |  |  |
| FM | IMU | ✓ |  |  |
| TUM, Replica, FM | GI-KF |  | ✓ |  |
| TUM, Replica, FM | Pyramid |  |  | ✓ |
| TUM, Replica, FM | GI-KF+Pyramid |  | ✓ | ✓ |
| FM | ALL | ✓ | ✓ | ✓ |

Replica uses a budget-matched keyframe cadence relative to its inherited baseline, and its transient support path is disabled. TUM and FMDataset enable the audited support policy. All other strategy semantics and formal parameters follow Section 3; in particular, the IMU setting is a rotation-only optimization contribution and the pyramid changes RGB-D supervision rather than the Gaussian representation.

**Selection and aggregation protocol.** The normalized evidence contains one fixed selected seed per scene and strategy: twenty TUM records, thirty-two Replica records, and eighteen FMDataset records. Every selected record uses seed 0. A strategy's macro value is the unweighted arithmetic mean of its scene values, with scenes rather than random seeds as the averaging units. The results are descriptive comparisons for this fixed selection; they are not random-seed averages, uncertainty estimates, or significance tests. All percentage changes use the corresponding Baseline value for the same scene or, for a stated dataset macro comparison, the Baseline macro over the identical scene set.

**Hardware and resource protocol.** TUM A1, A2, A3, and A5 and all FMDataset strategies were recorded on an NVIDIA GeForce RTX 2080 Ti; TUM A4 and all Replica strategies were recorded on an NVIDIA GeForce RTX 4080 SUPER. Every within-scene strategy comparison preserves the recorded hardware. We therefore use paired within-scene changes for resource interpretation and treat absolute values across different GPU models only as environment descriptions. Runtime is end-to-end SLAM elapsed time. Peak memory means the maximum PyTorch allocation returned by `torch.cuda.max_memory_allocated()`, not whole-device occupancy. Persistent-keyframe counts exclude transient tracking-support updates, whose counts and elapsed time are recorded separately.

**Metrics.** TUM and Replica tracking use root-mean-square absolute trajectory error (ATE), relative translation error (RPE-t), and relative rotation error (RPE-r). ATE is reported after rigid (SE(3)) alignment without scale correction [@horn1987_absoluteorientation]; lower values are better. FMDataset has no tracking result because the required reference trajectory is absent from the records.

Rendering uses a fixed list of sampled input observations under each method's own estimated trajectory. PSNR and MS-SSIM measure photometric and multiscale structural agreement, LPIPS measures perceptual distance, and Depth L1 measures valid-pixel rendered-depth error [@wang2003_msssim; @zhang2018_lpips]. PSNR and MS-SSIM are higher-is-better; LPIPS and Depth L1 are lower-is-better. The evaluated images are observed views rendered from the direct, unrefined submap concatenation. These measurements do not establish unseen-view generalization, independent global-map geometry, or mesh accuracy.

The detailed scene results are provided in [the TUM table](../../tables/tum-results.md), [the Replica table](../../tables/replica-results.md), [the FMDataset table](../../tables/fm-results.md), and [the paired resource table](../../tables/efficiency-results.md).
