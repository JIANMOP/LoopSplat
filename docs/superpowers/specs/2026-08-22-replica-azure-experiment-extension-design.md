# Replica 与 Azure 实验扩展设计

## 目标

将 Replica 八个标准场景纳入可复现的正式消融矩阵；按自采数据的真实文件结构修正 Azure Kinect RGB-D 预处理，但不把 Azure 纳入正式消融。全部适配通过本机与服务器 GPU 冒烟后，重写 `md/ablation-guide.md` 作为唯一实验执行手册。

## 冻结范围

- TUM RGB-D：5 场景，Baseline、GI-KF、Pyramid、GI-KF+Pyramid，3 seeds，共 60 runs。
- Replica：8 场景，Baseline、GI-KF、Pyramid、GI-KF+Pyramid，3 seeds，共 96 runs。
- FMDataset：3 场景，Baseline、IMU、GI-KF、Pyramid、GI-KF+Pyramid、ALL，3 seeds，共 54 runs。
- 正式矩阵合计 70 个配置、210 runs。
- Azure Kinect：只运行 Baseline GPU 冒烟并可在论文中提供定性图；不进入 `run_ablation.py`、汇总主表或策略统计。

## 数据契约

Replica 的真实目录为 `data/Replica/<scene>/`。每个场景必须包含 `results/frame*.jpg`、`results/depth*.png` 和 `traj.txt`，且三者数量一致。八个正式场景为 `office0`--`office4`、`room0`--`room2`。

Azure 场景为 `data/AzureKinect/144_5FPS_720p_IMU/`。`frame_info.json` 和 `imu.txt` 的时间戳单位均为秒。RGB 为 1280×720，深度为 640×576；使用 `camera_parameters.json` 中的深度/彩色内参、畸变和 `depth_to_color_transformation`，先校正畸变，再把深度几何注册到彩色相机，不允许仅按尺寸拉伸深度。

Azure 未提供 camera-to-IMU 外参且 IMU 采样率与图像约同为 5 Hz，因此 `tracking.use_imu=false`，不得将该场景作为 IMU 消融证据。

## 评测协议

Replica 正式主表使用与 TUM 相同的轨迹和固定 observed-view 协议：aligned ATE、RPE translation、RPE rotation、PSNR、SSIM、LPIPS、有效深度像素上的 Depth L1、关键帧数、子图数、SLAM 时间和峰值显存。

当前 Replica 数据没有 Co-SLAM mesh-culling 所需的 `gt_mesh_cull_virt_cams.ply` 与 `gt_pc_unseen.npy`，因此不启用或伪造 reconstruction mesh 指标。后续只有补齐标准裁剪资产后才能增加该表。

所有正式策略保持同一代码提交、GPU/CUDA、GSR 预算、评测帧和三个 seeds。产物完整性继续由 `formal_outputs_complete()` 验收。

## 输出与文档

Replica 输出沿用：

```text
output/ablation/R<scene>_<strategy>/seed_<seed>/<run_id>/
```

`md/ablation-guide.md` 必须删除旧的 Azure 消融、44配置和旧flat输出说明，写明210 runs、smoke命令、正式分组命令、断点续跑、产物验收、审计文件、汇总命令及论文表格解释。
