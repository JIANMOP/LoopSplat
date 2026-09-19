# Experiment audit

This is an internal provenance record, **not a submission attachment**. It defines the admissible experiment labels and reporting boundaries for the manuscript.

## Scene-to-ID mapping

| Group | Dataset | Scene IDs |
|---|---|---|
| A | TUM RGB-D | `A1=fr1/desk`; `A2=fr1/desk2`; `A3=fr1/room`; `A4=fr2/xyz`; `A5=fr3/long_office_household` |
| R | Replica | `R1=office0`; `R2=office1`; `R3=office2`; `R4=office3`; `R5=office4`; `R6=room0`; `R7=room1`; `R8=room2` |
| C | FMDataset | `C1=dorm1_fast1`; `C2=dorm2_fast`; `C3=hotel_fast1` |

Source: `scripts/run_ablation.py` scene definitions and `md/ablation-guide.md` §1.3.

## Strategy suffix mapping

| Groups | Suffix | Strategy | GI-KF | Pyramid | IMU |
|---|---|---|---:|---:|---|
| A, R | `_0` | Baseline | off | off | off |
| A, R | `_1` | +GI-KF | on | off | off |
| A, R | `_2` | +Pyramid | off | on | off |
| A, R | `_3` | +GI-KF+Pyramid | on | on | off |
| C | `_0` | Baseline | off | off | off |
| C | `_1` | +IMU | off | off | weak rotation-only |
| C | `_2` | +GI-KF | on | off | off |
| C | `_3` | +Pyramid | off | on | off |
| C | `_4` | +GI-KF+Pyramid | on | on | off |
| C | `_5` | +ALL | on | on | weak rotation-only |

For IMU-enabled C strategies, `lambda_imu_rot=0.001` and `lambda_imu_trans=0.0`; the configuration does not jointly estimate visual–inertial translation, velocity, bias, or gravity state. Replica GI-KF uses the budget-matched schedule in `scripts/run_ablation.py` `STRATEGIES_R`.

## Available metrics and reporting limits

| Dataset group | Admissible metrics | Exclusions |
|---|---|---|
| A, R | ATE, RPE, PSNR, SSIM, LPIPS, Depth L1, persistent keyframe count, submap count, run time, and peak GPU memory | Fixed-observed-view rendering must not be named novel-view synthesis; no mesh-reconstruction metrics are invented. |
| C | PSNR, SSIM, LPIPS, Depth L1, persistent keyframe count, submap count, run time, peak GPU memory, and separately reported tracking-support updates | No trajectory ground truth: do not report ATE/RPE or trajectory conclusions. |
| Azure Kinect | Qualitative engineering/useability examples only | Not a formal ablation; no ATE/RPE or quantitative IMU-validation conclusion. |

## Fixed-seed and hardware-comparison rule

The approved design states that the current results use the same fixed random seed for each configuration. Therefore, the manuscript may report per-scene values and cross-scene macro averages, but must not describe them as multi-seed means, report unavailable standard deviations, or claim statistical significance. The legacy three-seed language in `md/ablation-guide.md` is not evidence for the current results and is not a manuscript reporting license.

Compare strategies only when the data, evaluation frames, base parameters, strategy switches, and compatible recorded hardware/runtime environment match. Hardware configurations are reported as experimental context, not as cross-hardware performance comparisons. Do not compare elapsed time or peak memory across different GPU/CUDA environments as if they were a strategy effect.

## Internal-only provenance records from run manifests

For each included run, collect and retain internally: Git commit, experiment source fingerprint, configuration hash and experiment-configuration hash, command, seed, Git-clean status, GPU/CUDA environment, requested/effective feature flags, evaluation-frame IDs, and formal-run status. Retain associated `effective_features.yaml`, `run_statistics.yaml`, `gaussian_pyramid_summary.yaml`, `imu_summary.yaml`, `rendering_metrics_observed_view.json`, and, where ground truth exists, trajectory metric files. These records never enter the manuscript, tables, supplementary package, or submission attachment.
