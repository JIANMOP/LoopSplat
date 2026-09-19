## 5. Results and discussion

### 5.1 TUM RGB-D: efficiency gain with scene-dependent trajectory cost

The complete TUM results are listed in [Table 2](../tables/tum-results.md). At the macro level, GI-KF+Pyramid reduces the mean persistent-keyframe count from 1,690.8 to 610.4, a 63.9% reduction. Fixed-observed-view PSNR rises from 13.257 to 13.470 dB, SSIM from 0.4675 to 0.4807, LPIPS falls from 0.5611 to 0.5438, and Depth L1 falls from 0.3244 to 0.3136. This quality trend is accompanied by weaker trajectory accuracy: macro ATE increases from 3.309 to 5.887 cm, RPE-t from 0.0069 to 0.0128 m, and RPE-r from 0.441° to 0.582°.

The aggregate trajectory result needs scene-level qualification. On A2, GI-KF+Pyramid improves ATE from 3.795 to 3.322 cm and increases PSNR by 1.093 dB. On A4, its ATE is 1.395 cm compared with 1.264 cm for Baseline, while the keyframe count falls from 3,397 to 1,136. A1 and A5 show moderate trajectory degradation. A3 is the decisive failure case: GI-KF alone reaches 563.912 cm ATE versus 7.005 cm for Baseline. Adding the pyramid reduces that failure to 18.782 cm, but does not recover the baseline trajectory. Pyramid alone also records 18.960 cm on A3. The available logs demonstrate that the strategies respond differently on this sequence; they do not establish a unique physical cause, so we refrain from attributing the failure to a particular scene property.

GI-KF alone illustrates why averages must not be read in isolation. Its TUM macro rendering measures improve to 13.680 dB PSNR, 0.4857 SSIM, and 0.5356 LPIPS, while the A3 trajectory failure raises macro ATE to 114.970 cm. The result is useful for resource control and observed-view reconstruction, but its current interval and support policy does not provide uniformly reliable TUM odometry. The combined variant is more conservative in trajectory terms and is the preferred TUM tradeoff among the proposed variants, although it still trails Baseline ATE.

### 5.2 Replica: budget-matched keyframing is stable

Replica results appear in [Table 3](../tables/replica-results.md). The budget-matched GI-KF+Pyramid configuration lowers the macro keyframe count from 426.5 to 357.0, or 16.3%, without the TUM-style trajectory failure. Macro ATE improves from 0.292 to 0.262 cm and RPE-t from 0.000632 to 0.000594 m; RPE-r changes slightly from 0.0202° to 0.0209°. At the same time, PSNR increases from 28.168 to 28.606 dB, SSIM from 0.9328 to 0.9381, LPIPS decreases from 0.1164 to 0.1139, and Depth L1 decreases from 0.0291 to 0.0255.

The per-scene rows show that the macro result is not driven by a single room. GI-KF+Pyramid improves PSNR in seven of eight scenes and improves ATE in five. R1, R7, and R8 have slightly higher ATE than Baseline, and R7 is the only scene with lower PSNR. The largest combined depth improvement occurs on R5, where Depth L1 falls from 0.0490 to 0.0405 and PSNR rises by 0.997 dB.

The single-strategy rows clarify the interaction. Pyramid obtains the best macro ATE, 0.252 cm, but its LPIPS rises to 0.1220. GI-KF gives 0.271 cm ATE, 28.416 dB PSNR, and 0.1096 LPIPS while using 357.2 keyframes. The combined variant yields the highest macro PSNR and SSIM and the lowest Depth L1, although it does not inherit the best LPIPS or ATE from the individual modules. On Replica, the main conclusion is therefore a balanced quality–efficiency improvement rather than dominance on every metric.

### 5.3 FMDataset: inertial and mapping ablations

[Table 4](../tables/fm-results.md) reports fixed-observed-view quality for FMDataset. Because reference trajectories are unavailable, no conclusion about localization accuracy is drawn. The ALL configuration raises macro PSNR from 13.920 to 15.080 dB and SSIM from 0.5903 to 0.6107. LPIPS decreases from 0.7384 to 0.6722 and Depth L1 from 0.9657 to 0.7181, a 25.6% depth-error reduction. Persistent keyframes fall from 780.0 to 325.0, or 58.3%.

The complete configuration improves PSNR, SSIM, LPIPS, and depth over Baseline on each of C1–C3, although it is not best on every individual metric. On C3, for example, ALL reaches 16.186 dB PSNR and 0.3757 Depth L1, compared with 14.081 dB and 0.6774 for Baseline. On C1, GI-KF+Pyramid has slightly higher PSNR than ALL, 13.684 versus 13.619 dB, whereas ALL has lower LPIPS, 0.6772 versus 0.7166.

IMU alone produces mixed evidence. Its macro PSNR increases by 0.230 dB, LPIPS decreases by 2.6%, and Depth L1 decreases by 18.6%, but SSIM falls from 0.5903 to 0.5781. The scene rows explain the disagreement: PSNR increases on C1 and C2 but decreases from 14.081 to 13.730 dB on C3; LPIPS worsens on C1 but improves on C2 and C3. The rotation prior therefore contributes useful information in some measurements, yet it is not an across-metric improvement by itself.

The mapping modules account for most of the resource reduction. GI-KF alone lowers the C-group keyframe count by 58.5%, reduces macro time by 14.6%, and reduces peak memory by 49.6%; its PSNR rises by 0.591 dB, while Depth L1 changes only from 0.9657 to 0.9534. Pyramid alone leaves the keyframe count unchanged but reduces Depth L1 by 19.1%. Combining GI-KF and Pyramid gives 14.639 dB PSNR, 0.6178 SSIM, and 0.7478 Depth L1. Adding the IMU rotation term to form ALL further improves PSNR, LPIPS, and depth, while SSIM decreases from 0.6178 to 0.6107.

### 5.4 Time, memory, and persistent-map size

The paired measurements in [Table 5](../tables/efficiency-results.md) isolate strategy effects from GPU differences. On the five TUM scenes, GI-KF+Pyramid reduces persistent keyframes by 55.0%–66.6% and elapsed time by 9.0%–21.1%. Its peak-memory reduction ranges from 47.8% on A1 to 54.2% on A3. These savings are substantial, but Section 5.1 shows that they are purchased with scene-dependent trajectory error.

All Replica variants for a scene run on the same RTX 4080 SUPER environment. GI-KF+Pyramid reduces persistent keyframes by 14.8%–18.7% and time by 2.5%–20.0% across R1–R8. Macro peak memory changes only from 6.947 to 6.875 GiB, indicating that Replica's memory peak is not governed mainly by the modest reduction in persistent frames. This is consistent with a pipeline whose renderer and active Gaussian state remain resident even when fewer observations are retained.

All FMDataset configurations run on the RTX 2080 Ti. ALL reduces peak memory from 7.197 to 3.770 GiB and keyframes from 780.0 to 325.0 at the macro level. Its elapsed time is 2.576 hours versus 2.607 hours for Baseline, but the scene-level changes range from a 7.5% reduction on C1 to a 6.9% increase on C3. The IMU-only configuration increases time by 13.7% and memory by 7.5%, so the near-neutral ALL runtime reflects savings from GI-KF offsetting the added inertial and mapping work.

Persistent keyframe count and submap count should not be conflated. On Replica, every strategy has the same 33.25 macro submap count despite different keyframe budgets. On TUM, GI-KF alone increases the macro submap count from 20.2 to 46.6, while GI-KF+Pyramid records 20.6. Submap creation still follows inherited boundary logic, and reducing stored mapping frames does not guarantee fewer submaps.

### 5.5 Interactions and practical selection

![Figure 3. Dataset-level metric changes relative to Baseline. Positive values are plotted in the direction of improvement, so lower-is-better metrics are sign-adjusted.](../figures/metric-deltas.png)

Figure 3 shows that no single module explains all improvements. The pyramid has little effect on keyframe count, but it can materially improve depth, as on FMDataset. GI-KF is responsible for most persistent-map and memory savings, but its TUM trajectory response is scene-dependent. The inertial regularizer is deliberately weak and changes quality metrics in different directions. Their combination is strongest on FMDataset quality and is balanced on Replica, while TUM retains a trajectory tradeoff.

![Figure 4. Quality–efficiency comparison using macro values. Dataset-specific interpretation is required because hardware and available metrics differ.](../figures/quality-efficiency.png)

For deployments where memory and processing time are constrained, GI-KF+Pyramid or ALL is a reasonable operating point provided that the target motion pattern is validated first. For trajectory-critical TUM-like use, Baseline remains safer on the current evidence, and A3 should be treated as a blocking counterexample to any universal robustness claim. The Azure Kinect sequence can illustrate engineering operation, but without a reference trajectory it cannot resolve this limitation.

### 5.6 Validity boundaries

The study compares 70 controlled scene–strategy configurations, but each configuration has one retained run. Cross-scene macro averages characterize this experiment set and do not measure stochastic uncertainty. The observed-view rendering protocol evaluates reconstruction agreement at sampled input views, not generalization to unseen viewpoints. FMDataset supports an IMU quality ablation but no trajectory claim. Finally, the two GPU models prevent absolute time comparisons across differently hosted scenes; only same-scene deltas are used to discuss algorithmic efficiency.
