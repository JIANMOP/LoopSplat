# Reference audit

All entries below have a verified persistent identifier. `arXiv:` identifies an arXiv record; DOI and CVF/IEEE venue data identify the published record where listed. Later bibliography work must preserve these identifiers and verify full author, venue, and page metadata.

| Citation key | Work | Verified identifier | Status / intended use |
|---|---|---|---|
| `kerbl2023_3dgs` | *3D Gaussian Splatting for Real-Time Radiance Field Rendering* | arXiv:2308.04079; DOI: 10.1145/3592433 | Foundational 3DGS reference. |
| `keetha2024_splatam` | *SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM* | arXiv:2312.02126; CVPR 2024, pp. 21357–21366 | Gaussian RGB-D SLAM context. |
| `yan2024_gsslam` | *GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting* | arXiv:2311.11700; CVPR 2024, pp. 19595–19604 | Gaussian SLAM and coarse-to-fine context. |
| `yugay2023_gaussianslam` | *Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting* | arXiv:2312.10070 | Submap-based Gaussian SLAM context. |
| `huang2024_photoslam` | *Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras* | arXiv:2311.16728; CVPR 2024, pp. 21584–21593 | Image-pyramid and photorealistic-mapping context. |
| `xu2024_loopsplat` | *LoopSplat: Loop Closure by Registering 3D Gaussian Splats* | arXiv:2408.10154 | Inherited backend reference. |
| `campos2021_orbslam3` | *ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM* | arXiv:2007.11898; DOI: 10.1109/TRO.2021.3075644 | Visual–inertial and multi-map comparison context; not evidence for the current rotation-only configuration. |
| `sun2024_mm3dgsslam` | *MM3DGS SLAM: Multi-modal 3D Gaussian Splatting for SLAM Using Vision, Depth, and Inertial Measurements* | arXiv:2404.00923 | Multimodal Gaussian-SLAM related work. |
| `wu2025_vingsmono` | *VINGS-Mono: Visual-Inertial Gaussian Splatting Monocular SLAM in Large Scenes* | arXiv:2501.08286 | Visual–inertial Gaussian-SLAM related work. |
| `liu2025_gislam` | *GI-SLAM: Gaussian-Inertial SLAM* | arXiv:2503.18275 | **Preprint**; source of related IMU/keyframe ideas, not a published-paper citation claim. |
| `burt1981_fastfilter` | *Fast Filter Transform for Image Processing* | DOI: 10.1016/0146-664X(81)90092-7 | Classical Gaussian-pyramid/filter-transform reference; *Computer Graphics and Image Processing* 16(1):20–51. |
| `burt1983_pyramid` | *The Laplacian Pyramid as a Compact Image Code* | DOI: 10.1109/TCOM.1983.1095851 | Classical Gaussian/Laplacian pyramid reference; IEEE Trans. Commun. 31(4):532–540. |

Verification sources consulted on 2026-09-19: arXiv records for the 3DGS, SplaTAM, GS-SLAM, Gaussian-SLAM, Photo-SLAM, LoopSplat, ORB-SLAM3, MM3DGS SLAM, VINGS-Mono, and GI-SLAM entries; the CVPR 2024 proceedings for SplaTAM, GS-SLAM, and Photo-SLAM venue/page data; the IEEE record for ORB-SLAM3; and the IEEE Transactions on Communications record for Burt–Adelson. GI-SLAM is explicitly retained as an arXiv preprint.
