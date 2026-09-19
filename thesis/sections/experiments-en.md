## 4. Experimental setup

### 4.1 Datasets and scene groups

We evaluate the system on three indoor RGB-D datasets with different purposes. TUM RGB-D provides real sensor sequences and trajectory ground truth [@sturm2012_tumrgbd]. Five sequences are used: fr1/desk, fr1/desk2, fr1/room, fr2/xyz, and fr3/long_office_household, denoted A1–A5. Together they contain 8,454 processed frames. Replica provides synthetic indoor imagery and exact camera trajectories [@straub2019_replica]. We use all eight commonly adopted scenes, office0–office4 and room0–room2 (R1–R8), for 16,000 processed frames. FMDataset supplies RGB-D and synchronized inertial measurements recorded during handheld indoor motion [@zhu2019_fmdataset]. Its dorm1_fast1, dorm2_fast, and hotel_fast1 sequences form C1–C3 and contain 2,340 processed frames.

The three groups answer different questions. TUM tests the modifications on real RGB-D sequences with trajectory reference. Replica tests whether budget-matched keyframe deferral remains useful in stable synthetic scenes whose baseline already maps every five frames. FMDataset is the only formal group with IMU input and is therefore used for the inertial ablation. Its distributed data used in this experiment do not provide trajectory ground truth, so C-group results contain no ATE or RPE. Separately captured Azure Kinect material is reserved for qualitative engineering inspection and is not part of the quantitative ablation.

### 4.2 Strategy matrix

The Baseline is the inherited LoopSplat configuration with all three additions disabled. TUM and Replica evaluate Baseline, GI-KF, Pyramid, and GI-KF+Pyramid. FMDataset adds IMU and ALL configurations, where ALL enables IMU, GI-KF, and Pyramid. Table 1 summarizes the switches.

**Table 1. Ablation strategy matrix. A check mark means that the module is enabled.**

| Dataset group | Strategy | IMU rotation | GI-KF | Pyramid |
|---|---|---:|---:|---:|
| TUM, Replica, FM | Baseline |  |  |  |
| FM | IMU | ✓ |  |  |
| TUM, Replica, FM | GI-KF |  | ✓ |  |
| TUM, Replica, FM | Pyramid |  |  | ✓ |
| TUM, Replica, FM | GI-KF+Pyramid |  | ✓ | ✓ |
| FM | ALL | ✓ | ✓ | ✓ |

The GI-KF parameters are held fixed except for the cadence needed to match each dataset's baseline. TUM and FMDataset use minimum, stable, and maximum gaps of 2, 3, and 10 frames. Replica uses 5, 6, and 10 frames because its baseline mapping cadence is five frames; tracking-support updates are disabled for this group. The information-score weights, motion thresholds, support-update length, and pyramid schedule are those given in Section 3. The inertial variants use a rotation weight of 0.001 and a translational weight of zero.

### 4.3 Protocol and implementation

Each table entry reports a scene-strategy result, and the “Macro average” row is the unweighted arithmetic mean across scenes. The current reporting protocol does not estimate run-to-run variance or support significance tests. All strategy comparisons preserve the dataset frames, evaluation-view list, base parameters, and inherited LoopSplat functions; only the switches in Table 1 change.

Experiments were run on cloud-provisioned 22-GiB NVIDIA GeForce RTX 2080 Ti instances and 16-GiB RTX 4080 SUPER GPUs. All strategies for a given scene used the same recorded GPU environment. TUM A1, A2, A3, and A5 and all FMDataset runs used the 22-GiB RTX 2080 Ti environment; TUM A4 and all Replica runs used the RTX 4080 SUPER. Runtime and memory deltas are therefore interpreted within the same scene. Absolute timing across scenes on different GPU models is presented only as execution context.

The runner stores the effective configuration, feature summaries, evaluation metrics, timing, peak PyTorch allocation, and persistent-keyframe decisions. A completed run is included only when these outputs exist and all reported metrics are finite. The paper-facing extractor selected one complete record for each of the 70 scene–strategy configurations: 20 on TUM, 32 on Replica, and 18 on FMDataset. The resulting tables are generated from a normalized CSV so that prose and figures share the same values.

### 4.4 Metrics

For TUM and Replica, trajectory accuracy is measured by root-mean-square absolute trajectory error (ATE, reported in centimetres) after Horn rigid \(SE(3)\) alignment without scale. Relative translation error (RPE-t, metres) and relative rotation error (RPE-r, degrees) are RMSE values over consecutive frame pairs. Lower is better. FMDataset omits these metrics because no reference trajectory is available in the experiment set.

Rendering quality is evaluated on a fixed list of observed frames. PSNR and MS-SSIM measure pixel and multiscale structural agreement, while LPIPS measures perceptual distance [@wang2003_msssim; @zhang2018_lpips]. Higher PSNR/MS-SSIM and lower LPIPS indicate better agreement. Each run is rendered at its own estimated poses from the direct, unrefined concatenation of its submap Gaussians. The images are observations used by the sequence rather than held-out novel viewpoints, so these metrics measure internal reprojection consistency under the estimated trajectory rather than world-coordinate map accuracy or novel-view generalization. Depth L1 is the mean absolute rendered-depth error on valid evaluated pixels and is lower-is-better.

Efficiency is reported using end-to-end SLAM time in hours, peak PyTorch-allocated GPU memory in GiB as returned by torch.cuda.max_memory_allocated(), persistent-keyframe count, submap count, and the separately logged number and elapsed time of tracking-support updates. Keyframe count never includes support observations. For the main efficiency interpretation, percentage changes use the same-scene Baseline as the denominator.

### 4.5 Reporting conventions

Tables 2–4 report per-scene accuracy and quality, followed by unweighted cross-scene macro averages. Table 5 records paired efficiency measurements and the GPU used by each scene. Figure 3 shows metric changes relative to Baseline, while Figure 4 places quality changes beside keyframe and memory changes. A dataset-level average can hide a localized failure; accordingly, the discussion retains the TUM A3 regression and the mixed FMDataset IMU response rather than relying only on aggregate values.
