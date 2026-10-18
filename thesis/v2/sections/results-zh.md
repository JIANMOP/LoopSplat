## 4.2. 跟踪精度

**TUM RGB-D。** 非 A3 场景表明跟踪响应依赖于场景。GI-KF+Pyramid 在 A2 上相对 Baseline 降低 ATE，但 A1、A4 和 A5 的 ATE 更高；Pyramid 单独取得最低 A1 ATE，却未在整个组中保持这一行为。在 A2，GI-KF+Pyramid 将 ATE 从 3.795 改为 3.322 cm，将 PSNR 从 12.501 改为 13.594 dB，并将持久关键帧从 635 改为 286，即减少 54.96%。

A3 是必须报告的反例：Baseline 的对齐 ATE 为 7.005 cm，GI-KF 为 563.912 cm，GI-KF+Pyramid 为 18.782 cm；组合值仍比 Baseline 高 168.13%。这一失败排除了关于一般 TUM 跟踪改进或防止发散的主张。现有记录确立了结果，但未识别其物理原因。

**Replica。** 在八个场景上，GI-KF+Pyramid 将宏 ATE 从 0.292 改为 0.262 cm，即降低 10.30%；将 RPE-t 从 0.000632 改为 0.000594 m，即降低 6.10%；但 RPE-r 从 0.02019 升至 0.02092 degrees，即增加 3.61%。Pyramid 单独具有最低宏 ATE，因此组合策略并非每个指标上的跟踪最优项。预算匹配的 Replica 结果所显示的是：在这一合成组中，延后部分持久观测仍可与相近的跟踪行为兼容。

## 4.3. 渲染与深度重建

在 TUM 上，GI-KF+Pyramid 将持久关键帧减少 63.90%，同时相对 Baseline 将固定已观测视角宏 PSNR 改变 +0.213 dB、MS-SSIM 改变 +2.83%、LPIPS 改变 -3.07%、Depth L1 改变 -3.33%。这些渲染变化与上文 A3 跟踪失败并存，不能替代轨迹精度。

在 Replica 上，GI-KF+Pyramid 将持久关键帧减少 16.30%，将宏 PSNR 提高 0.438 dB，并将 Depth L1 降低 12.59%。其 RPE-r 高于 Baseline，Pyramid 有更低 ATE，GI-KF 有更低 LPIPS。因此，组合策略展示的是质量--预算权衡，而非跨指标占优。

FMDataset 只支持重建和资源主张。相对其 Baseline 宏，ALL 将固定已观测视角 PSNR 提高 1.159 dB、MS-SSIM 提高 3.47%，并将 LPIPS 降低 8.97%、Depth L1 降低 25.64%、持久关键帧降低 58.33%。记录中没有轨迹参考，因此这些差异不证明定位、漂移或视觉惯性里程计得到改善。

![相对 Baseline 的数据集级指标变化。仅为可视化，对数值越低越好的量作符号调整。](../../figures/metric-deltas.png){#fig:metric-deltas}

## 4.4. 资源效率

配对资源测量将策略变化与硬件差异分离。在 Replica 上，GI-KF+Pyramid 将宏耗时降低 11.94%，同场景降幅介于 2.5% 至 20.0%；宏峰值 PyTorch 分配量从 6.947 改为 6.875 GiB。很小的分配量变化说明，持久观测数量并非常驻 Replica 工作负载的唯一组成。

在 FMDataset 上，ALL 将宏峰值 PyTorch 分配量降低 47.61%，将宏耗时降低 1.18%。其逐场景耗时变化为 C1 的 -7.5%、C2 的 -0.6% 和 C3 的 +6.9%，所以该宏结果不是一致的逐场景加速。所有这些比较均在同一 RTX 2080 Ti 环境中进行，不构成跨 GPU 基准。

持久存储与更新核算也必须分开。TUM 组合宏包含 610.4 个持久关键帧和 1080.4 次临时支撑更新，共处理 1690.8 帧。因此，报告的持久关键帧减少并不意味着建图或更新事件等比例减少：支撑更新在实时地图上执行更短、非持久的优化，并单独计时。

![固定已观测视角质量与资源变化。数值具有数据集特异性，不应作为绝对的跨硬件性能比较。](../../figures/quality-efficiency.png){#fig:quality-efficiency}

## 4.5. 消融研究

**惯性旋转。** 在 FMDataset 上，仅 IMU 将宏 PSNR 改变 +0.230 dB、Depth L1 改变 -18.55%、LPIPS 改变 -2.56%，但将 MS-SSIM 改变 -2.06%；其耗时和峰值 PyTorch 分配量分别增加 13.71% 和 7.45%。它还使 C1 LPIPS 和 C3 PSNR 变差。该组件因而给出混合的重建证据，并具有可测量的成本。由于 FMDataset 缺少轨迹真值，此消融无法证明定位或漂移收益。

**观测自适应关键帧。** FMDataset GI-KF 将持久关键帧减少 58.50%，并将 PSNR 提高 0.591 dB，而相对 Baseline 的 Depth L1 仅降低 1.27%。这将 GI-KF 的主要证据限定为持久观测与资源控制，并不支持无代价重建主张。Replica 的行中支撑被禁用且节奏预算匹配，进一步表明接纳规则可不依赖临时更新而减少持久集合。

**按使用次数推进的粗到细监督。** FMDataset Pyramid 保持宏持久关键帧数量不变，将 PSNR 提高 0.081 dB，并将相对 Baseline 的 Depth L1 降低 19.12%。这将监督调度与接纳分开：其最清晰的单组件信号是深度重建，而非存储减少。逐场景结果仍然混合，不能由此推断金字塔会改善每个序列或指标。

**兼容性。** FMDataset GI-KF+Pyramid 将宏 PSNR 提高 0.718 dB、MS-SSIM 提高 4.67%，将 LPIPS 降低 5.28%、Depth L1 降低 22.56%，并将相对 Baseline 的持久关键帧降低 58.50%。启用惯性旋转后，ALL 再次改变这一平衡，改善部分宏指标，但 MS-SSIM 低于 GI-KF+Pyramid。这些组合表明决策能够共同运行；它们并不说明每个组件会改善每项指标。

## 4.6. 定性与失败分析

Replica R5 建图器面板展示了同一已观测帧在 Baseline 与 GI-KF+Pyramid 下的采样输入、渲染 RGB-D 输出、深度残差和增密掩码。它们是优化诊断，而非留出渲染或完整全局地图评测。

![已观测帧 1990 上的 Replica R5 Baseline 建图器诊断。](../../figures/qualitative/replica-r5-baseline.jpg){#fig:r5-baseline}

![已观测帧 1990 上的 Replica R5 GI-KF+Pyramid 建图器诊断。](../../figures/qualitative/replica-r5-combined.jpg){#fig:r5-combined}

A3 轨迹可视化了第 4.2 节已量化的失败。Baseline 保持在对齐参考附近，而 GI-KF 显著偏离。这一对照说明，固定已观测视角的渲染一致性不能代替全局轨迹精度，也说明在轨迹敏感设置中，观测接纳仍需要更安全的回退机制。

![刚性对齐后的 TUM A3 Baseline 轨迹。](../../figures/qualitative/tum-a3-baseline-trajectory.png){#fig:a3-baseline}

![刚性对齐后的 TUM A3 GI-KF 轨迹。](../../figures/qualitative/tum-a3-gikf-trajectory.png){#fig:a3-gikf}

失败边界来自经验结果。记录没有分离运动、重叠估计、地图状态或其他因素是否导致 A3，因此归因于某种物理机制将超出证据。类似地，没有受控参考的 Azure Kinect 材料仍是工程说明，而非消融或精度结果。
