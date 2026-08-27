# GI-SLAM: Gaussian-Inertial SLAM

Xulang Liu Sun Yat-sen University Guangzhou, China

Ning Tan Sun Yat-sen University Guangzhou, China

# Abstract

3D Gaussian Splatting (3DGS) has recently emerged as a powerful representation of geometry and appearance for dense Simultaneous Localization and Mapping (SLAM). Through rapid, differentiable rasterization of 3D Gaussians, many 3DGS SLAM methods achieve near realtime rendering and accelerated training. However, these methods largely overlook inertial data, witch is a critical piece of information collected from the inertial measurement unit (IMU). In this paper, we present GI-SLAM, a novel gaussian-inertial SLAM system which consists of an IMU-enhanced camera tracking module and a realistic 3D Gaussian-based scene representation for mapping. Our method introduces an IMU loss that seamlessly integrates into the deep learning framework underpinning 3D Gaussian Splatting SLAM, effectively enhancing the accuracy, robustness and efficiency of camera tracking. Moreover, our SLAM system supports a wide range of sensor configurations, including monocular, stereo, and RGBD cameras, both with and without IMU integration. Our method achieves competitive performance compared with existing state-of-the-art real-time methods on the EuRoC and TUMRGBD datasets.

still face substantial limitations for applications requiring high-fidelity, fine-grained 3D dense mapping.

To enhance the visual fidelity of 3D maps, researchers have explored SLAM methods based on Neural Radiance Fields (NeRF)[22]. These approaches leverage differentiable volumetric rendering to train implicit neural networks, enabling photometrically accurate 3D representations of the environment. This novel map representation is compact, continuous, and efficient, and it can be optimized through differentiable rendering, making it particularly beneficial for applications such as navigation, planning, and reconstruction. Methods such as iMAP[33] pioneered the use of MLP-based representations for realtime tracking and mapping, while NICE-SLAM[48] introduced a hierarchical voxel-based strategy to enhance scalability and reconstruction quality. Further advancements like Vox-Fusion[45] and ESLAM[15] have improved memory efficiency and tracking robustness through dynamic voxel allocation and multi-scale feature planes. Co-SLAM[37] explore InstantNGP[24] to further accelerate the mapping speed. Despite these advancements, NeRF-SLAM still faces challenges such as high computational cost, slow convergence, and difficulties in handling large-scale dynamic environments due to model capacity limitations and catastrophic forgetting.

# 1. Introduction

Dense Visual Simultaneous Localization and Mapping (SLAM) is a fundamental problem in 3D computer vision with numerous applications in fields such as autonomous driving, robotic navigation and planning, as well as augmented reality (AR) and virtual reality (VR). The goal of dense SLAM is to construct a detailed map of an unknown environment while simultaneously tracking the camera pose, ensuring that embodied agents can perform downstream tasks in previously unseen 3D environments. Traditional SLAM methods utilize point clouds[9, 25, 44], surfels[38], meshes[1], or voxel grids[23] for scene representation to create dense maps, achieving significant progress in localization accuracy. However, these methods

More recently, 3D Gaussians[18] have emerged as an alternative representation of radiance fields, achieving equal or superior rendering quality compared to traditional NeRFs while being significantly faster in rendering and training. Building on this success, some works[12, 16, 21, 41, 43] have integrated 3D Gaussian Splatting (3DGS) with dense visual SLAM systems and achieved significant advancements in both mapping accuracy and computational efficiency. However, to the best of our knowledge, no existing work has effectively integrated inertial data—which is crucial for accurate camera pose estimation in visual SLAM—into a 3DGS SLAM framework.

IMUs are commonly found in many mobile devices and robots, and they are frequently integrated into traditional visual SLAM systems[2, 4, 5, 29, 39] to enhance the accuracy and robustness of camera tracking. To this end, we propose GI-SLAM, a novel Gaussian-inertial SLAM system that consists of an IMU-enhanced camera tracking module and a photo-realistic 3D Gaussian scene representation for mapping. In our system, we introduce a novel IMU loss function, which, when combined with the photometric loss function, improves the accuracy and robustness of camera tracking. Additionally, we propose a keyframe selection strategy based on Gaussian visibility and motion data to prevent the selection of motion-blurred keyframes, thereby improving the quality of the reconstructed map. In summary, the main contributions of this work include:

• We propose a novel simultaneous localization and photorealistic mapping system based on 3D Gaussian Splatting (3DGS), called GI-SLAM. The proposed framework supports monocular, stereo, and RGBD cameras, with or without IMU integration, and is designed to operate effectively in both indoor and outdoor environments. • We present a keyframe selection strategy based on Gaussian visibility and motion constraint to prevent the selection of motion-blurred keyframes. •We conducted extensive evaluations of our proposed system on the TUM and EuRoC datasets with IMU data, demonstrating competitive performance.

# 2. Related Work

In this section, we provide a concise overview of various dense SLAM approaches, with a particular focus on recent methods that utilize implicit representations encoded in overfit neural networks for tracking and mapping. For a more comprehensive review of NeRF-based and 3DGSbased SLAM techniques, we refer interested readers to the excellent survey[36].

Classical dense SLAM There has been extensive work on 3D reconstruction over the past decades. Traditional dense SLAM methods have explored various map representations, including point clouds[9, 17, 19], surfels[31, 38, 42], and Gaussian mixture models[10, 11]. truncated signeddistance functions (TSDF)[6, 7, 14, 26]. Among these, Surfel-based approaches like ElasticFusion[42] represent the scene as a collection of circular surface elements, enabling real-time tracking and mapping. TSDF-based methods such as KinectFusion[26] and its extensions[27, 28] have demonstrated impressive real-time 3D reconstruction capabilities by leveraging efficient volumetric integration and voxel hashing techniques. While these classical methods provide geometrically accurate reconstructions, they often struggle with scalability and handling uncertainty in depth measurements. Recent advances have introduced learning-based methods such as RoutedFusion[40] and DIFusion[13], which leverage deep networks to improve robustness against noisy depth inputs. These works primarily focus on the geometry reconstruction, while differently, our method takes both 3D reconstruction and photorealistic rendering into account simultaneously.

NeRF-based dense SLAM Several recent works have explored the use of neural radiance fields(NeRF)[22] for dense SLAM, leveraging its ability to efficiently represent scene geometry and appearance. For instance, iMAP[33] pioneered the integration of neural implicit representations into SLAM, using a single MLP to dynamically build a scenespecific 3D model. This approach provides efficient geometry representation and automatic detail control but struggles with scalability and catastrophic forgetting in larger environments. NICE-SLAM[48] improved upon iMAP by adopting a hierarchical feature grid representation with multi-level voxel encodings, enhancing reconstruction quality and mitigating catastrophic forgetting through localized updates. Vox-Fusion[45] further introduced an octree-based voxel allocation strategy, allowing for dynamic scene encoding and efficient memory management. More recent advancements, such as ESLAM[15], utilize multi-scale feature planes to optimize memory efficiency and improve reconstruction speed by leveraging a TSDF-based representation. Co-SLAM[37] combines smooth coordinate encodings with sparse hash grids, achieving robust tracking and high-fidelity map reconstruction with efficient hole-flling techniques. Additionally, GO-SLAM[47] integrates global optimization techniques like loop closure and bundle adjustment to ensure long-term trajectory consistency. Alternative representations have also been explored. Point-SLAM[30] employs dynamic neural point clouds, adaptively adjusting density based on scene information for memory-efficient mapping. Other works such as Plenoxel-SLAM[35] eschew neural networks entirely, leveraging voxel grids with trilinear interpolation for efficient real-time mapping and tracking. Despite these advances, NeRF-based SLAM systems face challenges related to computational complexity and memory consumption, which hinder real-time performance on large-scale scenes. Our 3DGS-based approach aims to address these limitations by introducing a more memoryefficient representation and faster rendering pipeline.

3DGS-based dense SLAM Recently, with the great success of 3D Gaussian Splatting(3DGS)[18], some works have integrated 3DGS with dense SLAM systems. MonoGS[21] employs 3D Gaussians as the sole representation for online reconstruction, utilizing direct optimization for camera tracking and introducing Gaussian shape regularization to maintain geometric consistency. PhotoSLAM[12] integrates explicit geometric features with implicit texture representations within a hyper primitives map, optimizing camera poses through multi-threaded factor graph solving. SplaTAM[16] refines tracking and mapping by optimizing Gaussian parameters through rerendering errors and incremental densification strategies, though it remains sensitive to motion blur and depth noise. GS-SLAM[43] introduces an adaptive expansion strategy to dynamically manage 3D Gaussians and a coarse-tofine tracking approach to enhance pose estimation accuracy. Gaussian-SLAM[46] organizes scenes into independently optimized sub-maps based on camera motion, improving scalability for larger environments. CompactGSSLAM[20] addresses memory constraints by reducing Gaussian ellipsoid parameters and employing a sliding window-based masking strategy for efficient resource allocation. Despite these advancements, challenges persist in achieving robust real-time performance, particularly in handling dynamic scenes and motion blur. Our approach introduces an IMU-based loss to incorporate inertial information into the 3DGS framework, significantly enhancing camera tracking accuracy and robustness. Additionally, motion constraints are utilized to filter frames susceptible to motion blur, thereby improving overall mapping quality. This integration of IMU data with 3DGS-based SLAM ensures more stable tracking and higher-fidelity reconstructions in challenging environments.

![](https://gitee.com/JianMop/picgo_-cloud/raw/master/20251204141917489.jpg)  
Figure 1. SLAM system overview.

# 3. Method

GI-SLAM contains three main components, including localization, mapping, and keyframing, shown in Fig. 1. Our proposed framework is an online SLAM system that simultaneously tracks camera poses and reconstructs dense scene geometry using RGB, depth, and IMU data. This is achieved through the following steps: First, the camera pose is initialized, and an initial 3D Gaussian scene representation is generated. Camera tracking is then performed by comparing the rendered RGB and depth images from the 3D Gaussian map with the input images, depth data, and IMU measurements. Each incoming frame is evaluated to determine whether it qualifies as a keyframe. If identified as a keyframe, the 3D Gaussian map is updated accordingly, which in turn enhances subsequent camera tracking. This process repeats iteratively until the SLAM system is terminated.

# 3.1. Localization

We take advantage of the differentiable nature of the 3D Gaussian map rendering process by parameterizing the pose matrix used to represent the camera pose, enabling optimization via gradient descent. The camera pose is represented using an $S E ( 3 )$ homogeneous transformation matrix:

$$
\mathbf { P } _ { t } \in S E ( 3 ) = \bigg [ \mathbf { R } _ { t } \quad \mathbf { t } _ { t } \bigg ] .
$$

We assume that the previous camera pose is accurate, and update the current camera pose based on it:

$$
\mathbf P _ { t } = \mathbf P _ { t - 1 } \cdot \Delta \mathbf P ,
$$

where $\mathbf { P } _ { t }$ represents the current camera pose, $\mathbf { P } _ { t - 1 }$ denotes the previous pose, and $\Delta \mathbf { P }$ is the incremental transformation estimated through optimization. The refinement of $\Delta \mathbf { P }$ is achieved by minimizing the alignment error between the sensor data (RGB, depth, and IMU measurements) and the corresponding RGB and depth images rendered from the 3D Gaussian map. This optimization process involves minimizing a weighted combination of three loss functions: the RGB loss, which measures photometric consistency; the depth loss, which ensures geometric accuracy; and the IMU loss, which enforces inertial constraints.

![](https://gitee.com/JianMop/picgo_-cloud/raw/master/20251204141917491.jpg)  
Figure 2. Trajectory tracking result of the monocular camera setup on the TUM dataset.

RGB Loss In the monocular case, we minimise the following RGB loss:

$$
\begin{array} { r } { \mathcal { L } _ { r g b } = \left\| { I ( \mathcal { G } , { \mathbf { P } } _ { t - 1 } \cdot \Delta { \mathbf { P } } ) - \bar { I } } \right\| _ { 1 } . } \end{array}
$$

This is an L1 loss function, where $I ( { \mathcal { G } } , \mathbf { P } _ { t - 1 } \cdot \Delta \mathbf { P } )$ represents the process of rendering the 3D Gaussians $\mathcal { G }$ into a 2D image at pose $\mathbf { P } _ { t - 1 } \cdot \Delta \mathbf { P }$ , and $\bar { I }$ denotes the ground truth image.

Depth Loss When depth observations are available, we define the depth loss as:

$$
\begin{array} { r } { \mathcal { L } _ { d e p t h } = \left. D ( \boldsymbol { \mathcal { G } } , \mathbf { P } _ { t - 1 } \cdot \Delta \mathbf { P } ) - \bar { D } \right. _ { 1 } , } \end{array}
$$

where $D ( \mathcal { G } , \mathbf { P } _ { t - 1 } \cdot \Delta \mathbf { P } )$ represents depth rasterisation, and $\bar { D }$ denotes the observed depth.

IMU Loss Our IMU loss function incorporates both translational and rotational constraints from 6-DOF IMU measurements. For translational constraints, we integrate the linear acceleration measurements $\mathbf { a } _ { t }$ with the previous camera frame's linear velocity $\mathbf { v } _ { t - 1 }$ :

$$
\Delta { \bf p } _ { i m u } = { \bf v } _ { t - 1 } \Delta t + \frac { 1 } { 2 } { \bf a } _ { t } \Delta t ^ { 2 } .
$$

The translation loss $\mathcal { L } _ { t r a n s }$ is then computed as:

where $\Delta \mathbf { p } _ { o p t } \in \mathbb { R } ^ { 3 }$ denotes the optimized displacement between consecutive frames.

For rotational constraints, we derive the relative rotation from angular velocity measurements $\omega _ { t }$ :

$$
\Delta \theta _ { i m u } = \omega _ { t } \Delta t .
$$

The rotation loss $\mathcal { L } _ { \mathit { r o t } }$ is formulated as:

$$
\mathcal { L } _ { r o t } = \| \Delta \theta _ { o p t } - \Delta \theta _ { i m u } \| _ { 2 } ^ { 2 } ,
$$

where $\Delta \theta _ { o p t } \in \mathbb { R } ^ { 3 }$ represents the optimized relative rotation in axis-angle form. The final IMU loss combines both components through weighted summation:

$$
\begin{array} { r } { \mathcal { L } _ { i m u } = \lambda _ { t } \mathcal { L } _ { t r a n s } + \lambda _ { r } \mathcal { L } _ { r o t } , } \end{array}
$$

with $\lambda _ { t }$ and $\lambda _ { r }$ being hyper-parameters balancing the two constraints. Both $\Delta \mathbf { p } _ { o p t }$ and $\Delta \theta _ { o p t }$ are optimized variables in our SLAM framework.

# 3.2. Mapping

Our method employs 3D Gaussians as the sole 3D scene representation to model the environment, capable of rendering precise RGB and depth images through differentiable rendering. This facilitates the use of gradient-based optimization techniques to update the 3D environmental map.

3D Gaussian Scene Representation Similar to MonoGS[21], our SLAM system models the scene as a collection of anisotropic 3D Gaussians $\begin{array} { r c l } { \mathcal { G } } & { = } & { \{ G _ { i } \} } \end{array}$ , where each Gaussian $G _ { i }$ is parameterized by its position $\pmb { \mu } _ { W } ^ { i } \in \mathbb { R } ^ { 3 }$ , covariance $\beth _ { W } ^ { i } \in \mathbb { R } ^ { 3 \times 3 }$ (defining ellipsoidal geometry), color $c ^ { i } \in \ \dot { \mathbb { R } } ^ { 3 }$ , and opacity $\alpha ^ { i } \in \mathbb { R }$ . To balance expressiveness and computational efficiency, we parameterize the covariance matrix as:

$$
\mathcal { L } _ { t r a n s } = \| \Delta \mathbf { p } _ { o p t } - \Delta \mathbf { p } _ { i m u } \| _ { 2 } ^ { 2 } ,
$$

$$
\pmb { \Sigma } _ { W } ^ { i } = \mathbf { R } _ { i } \mathbf { S } _ { i } \mathbf { S } _ { i } ^ { \top } \mathbf { R } _ { i } ^ { \top } ,
$$

where $\mathbf { S } _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ is a diagonal scaling matrix, and $\mathbf { R } _ { i } \in$ SO(3) (represented as a quaternion) defines rotation. This formulation avoids explicit surface extraction and enables efficient scene reconstruction through volume rendering. The continuous 3D representation allows adaptive refinement of Gaussians during mapping.

Differentiable Rendering To bridge 3D Gaussians with 2D observations, we project $G _ { i }$ onto the image plane via a differentiable geometric transformation. Given camera pose $\pmb { T } _ { C W } \in \mathrm { S E } ( 3 )$ , the 3D Gaussian $\mathcal { N } ( \pmb { \mu } _ { W } , \pmb { \Sigma } _ { W } )$ maps to a 2D Gaussian $\mathcal { N } ( \mu _ { I } , \Sigma _ { I } )$ as:

$$
\begin{array} { r } { \pmb { \mu } _ { I } = \pi ( \mathbf { T } _ { C W } \pmb { \mu } _ { W } ) , \quad \pmb { \Sigma } _ { I } = \mathbf { J } \mathbf { W } \pmb { \Sigma } _ { W } \mathbf { W } ^ { \top } \mathbf { J } ^ { \top } , } \end{array}
$$

where $\pi$ is the perspective projection, $\mathbf { W }$ is the rotational component of ${ \pmb T } _ { C W }$ , and $\mathbf { J }$ is the Jacobian of the projective approximation.

Pixel colors $\hat { \mathbf { C } }$ and depths $\hat { D }$ are rendered via front-toback $\propto$ blending:

$$
\hat { \mathbf { C } } = \sum _ { i \in \mathcal { N } } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,
$$

$$
\hat { D } = \sum _ { i \in \mathcal { N } } d _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,
$$

where $\alpha _ { i }$ is modulated by the projected 2D Gaussian's density, and $d _ { i }$ is the depth of $G _ { i }$ 's centroid in the camera frame. Crucially, this rendering process is fully differentiable, enabling gradient-based optimization of Gaussian parameters.

Map Update The 3D Gaussian scene representation is dynamically refined through a continuous cycle of initialization, adaptive density control, and gradient-driven optimization. Initial Gaussians are seeded from sparse geometric cues, such as keyframe-based triangulation or depth estimation, where their initial positions and covariances reflect the uncertainty of sensor observations. To ensure the scene is neither under- nor over-reconstructed, the system employs a density adaptation mechanism: Gaussians are cloned in regions with high photometric reconstruction errors (indicating missing geometry) and pruned if their opacity falls below a minimal threshold or their spatial extent becomes negligible (e.g., due to occlusion or excessive compression). This balance allows the map to grow in complex regions while eliminating redundant elements.

The core of the update process lies in differentiable optimization. We formulate an L1 composite loss function $\mathcal { L }$ to align the rendered outputs with sensor observations, defined as:

$$
\mathcal { L } = \| \hat { \mathbf { C } } - \mathbf { C } _ { \mathrm { g t } } \| _ { 1 } + \lambda \| \hat { D } - D _ { \mathrm { g t } } \| _ { 1 } ,
$$

where $\hat { \mathbf { C } }$ and $\hat { D }$ denote the rendered RGB image and depth map respectively, while $\mathbf { C } _ { \mathrm { g t } }$ and $D _ { \mathrm { g t } }$ represent the corresponding ground-truth measurements captured by the sensors. The weighting coefficient $\lambda$ balances the photometric and geometric error terms. Through backpropagation of this differentiable loss function, we jointly optimize a set of 3D Gaussian scene representation parameters including positions $\pmb { \mu } _ { W }$ , covariance scales S, rotations $\mathbf { R }$ ,colors $c$ , and opacities $\alpha$ . The optimization adjusts Gaussians to better align their projected 2D contributions with observed colors and depths, while implicit regularization maintains physically plausible shapes. Through iterative updates, the Gaussian ensemble converges to a compact yet accurate representation of the scene's geometry and appearance.

# 3.3. Keyframing

Jointly optimizing all video stream frames with 3D Gaussians and camera poses in real-time is computationally infeasible. Inspired by MonoGS[21], we maintain a compact keyframe window $\mathcal { W } _ { k }$ , dynamically curated through interframe covisibility analysis. Ideal keyframe management balances two objectives: (1) selecting non-redundant and high-quality frames observing overlapping scene regions to ensure multiview constraints, and (2) maximizing baseline diversity between keyframes to strengthen geometric stability. This strategy minimizes redundancy while preserving reconstruction accuracy, as detailed in our supplementary material.

Selection and Management Keyframe selection is governed by a unified scoring function that consolidates three critical criteria: covisibility, baseline span, and motion stability. For each tracked frame $i$ , the keyframe score $s _ { i }$ is computed as:

$$
\begin{array} { r l } & { s _ { i } = w _ { \mathrm { c o v i s } } \cdot ( 1 - \mathrm { I o U } _ { \mathcal { G } } ) } \\ & { ~ + w _ { \mathrm { b a s e } } \cdot \frac { \| t _ { i j } \| } { d _ { \mathrm { m e d } } } } \\ & { ~ - w _ { \mathrm { m o t } } \cdot \mathbb { I } \left( v _ { i } > v _ { \mathrm { m a x } } \lor \omega _ { i } > \omega _ { \mathrm { m a x } } \right) , } \end{array}
$$

where $\operatorname { I o U } _ { \mathcal { G } }$ quantifies the overlap of visible 3D Gaussians between the current frame and the latest keyframe, $\| t _ { i j } \| / d _ { \mathrm { m e d } }$ measures the normalized baseline span, and $v _ { i } , \omega _ { i }$ denote linear and angular velocities. The weights $w _ { \mathrm { c o v i s } }$ $w _ { \mathrm { b a s e } }$ , and $w _ { \mathrm { m o t } }$ balance these factors, while $\mathbb { I } ( \cdot )$ enforces motion constraints.

A frame is selected as a keyframe if $s _ { i } \ > \ \tau _ { \mathrm { k f } }$ (e.g., $\tau _ { \mathrm { k f } } = 0 . 7 )$ and the keyframe window $\mathcal { W } _ { k }$ has capacity. To maintain efficiency, redundant keyframes are pruned when their overlap with the latest keyframe exceeds $\tau _ { \mathrm { o v e r l a p } }$ (e.g., 0.8). This strategy ensures a compact set of non-redundant keyframes that maximize multiview constraints while minimizing computational overhead.

Table 1. Camera tracking results on TUM for monocular and RGBD(ATE RMSE ↓ [cm]).   

<table><tr><td>Input</td><td>Methods</td><td>fr1/desk</td><td>fr2/xyz</td><td>fr3/office</td><td>Avg.</td><td>fr1/desk2</td><td>fr1/room</td><td>Avg.</td></tr><tr><td rowspan="4">Monocular</td><td>DROID-VO[34]</td><td>5.12</td><td>9.88</td><td>7.30</td><td>7.43</td><td>b</td><td>-</td><td>-</td></tr><tr><td>DepthCov-VO[8]</td><td>5.63</td><td>1.20</td><td>53.4</td><td>20.08</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MonoGS[21]</td><td>3.56</td><td>4.59</td><td>3.50</td><td>3.88</td><td>77.64</td><td>79.88</td><td>79.36</td></tr><tr><td>Ours</td><td>1.98</td><td>3.27</td><td>3.14</td><td>2.80</td><td>50.27</td><td>61.43</td><td>55.85</td></tr><tr><td rowspan="6">RGBD</td><td>NICE-SLAM[48]</td><td>4.24</td><td>6.04</td><td>3.85</td><td>4.71</td><td>4.89</td><td>33.79</td><td>19.34</td></tr><tr><td>Vox-Fusion[45]</td><td>3.48</td><td>1.53</td><td>23.7</td><td>9.57</td><td>6.00</td><td>25.73</td><td>15.87</td></tr><tr><td>Point-SLAM[30]</td><td>4.11</td><td>1.43</td><td>3.19</td><td>2.91</td><td>4.54</td><td>33.94</td><td>19.24</td></tr><tr><td>SplaTAM[16]</td><td>3.34</td><td>1.22</td><td>5.20</td><td>3.25</td><td>6.54</td><td>11.10</td><td>8.82</td></tr><tr><td>MonoGS[21]</td><td>1.59</td><td>1.36</td><td>1.55</td><td>1.50</td><td>6.30</td><td>6.64</td><td>6.47</td></tr><tr><td>Ours</td><td>1.34</td><td>1.26</td><td>1.54</td><td>1.38</td><td>4.82</td><td>4.66</td><td>4.74</td></tr></table>

Gaussian Covisibility Covisibility is estimated using the visibility properties of 3D Gaussians during differentiable rendering. A Gaussian $G _ { k }$ is deemed visible in frame $i$ if:

•Its contribution to pixel colors exceeds a minimal threshold ( $\alpha$ -blending weight $> 0 . 0 1$ ),   
The accumulated opacity along its ray does not surpass 0.5, implicitly handling occlusions.

The covisibility between frames $i$ and $j$ is defined as:

$$
\mathrm { I o U } _ { \mathcal { G } } = \frac { | \mathcal { G } _ { i } \cap \mathcal { G } _ { j } | } { | \mathcal { G } _ { i } \cup \mathcal { G } _ { j } | } ,
$$

where $\mathcal { G } _ { i }$ and $\mathcal { G } _ { j }$ are sets of visible Gaussians. This metric avoids explicit geometric overlap checks and naturally adapts to scene complexity, providing a robust measure of shared scene content.

Motion Data Constraint To mitigate motion blur, frames with excessive linear velocity $( \pmb { v } _ { i } > \tau _ { \operatorname* { m a x } } )$ or angular velocity $\left( \omega _ { i } > \omega _ { \operatorname* { m a x } } \right)$ are excluded from keyframe candidacy. Thresholds $v _ { \mathrm { m a x } }$ and $\omega _ { \mathrm { m a x } }$ are empirically determined based on sensor characteristics (e.g., camera frame rate and IMU noise). This hard rejection rule ensures only geometrically stable frames are retained, enhancing mapping quality without compromising real-time performance.

# 4. Experiments

We conducted extensive experiments to evaluate the performance of GI-SLAM in terms of both camera tracking accuracy and mapping quality. Our experiments were designed to benchmark against state-of-the-art methods and assess the effectiveness of GI-SLAM in diverse settings, including different datasets and evaluation metrics.

# 4.1. Experimental Setup

Implementation details GI-SLAM is implemented in Python using the PyTorch framework, incorporating CUDA code for Gaussian splatting. The training and evaluation were performed on a desktop PC equipped with a $6 . 0 \mathrm { G H z }$ Intel Core i9-14900K CPU and an NVIDIA RTX 4090 GPU. More technical details can be found in the supplemental materials.

Table 2. Rendering performance on TUM for RGBD.   

<table><tr><td>Methods</td><td>Metric</td><td>fr1/desk</td><td>fr2/xyz</td><td>fr3/office</td><td>Avg.</td></tr><tr><td rowspan="3">NICE-SLAM[48]</td><td>PSNR↑</td><td>13.87</td><td>17.94</td><td>15.11</td><td>15.64</td></tr><tr><td>SSIM↑</td><td>0.566</td><td>0.668</td><td>0.561</td><td>0.598</td></tr><tr><td>LPIPS↓</td><td>0.485</td><td>0.327</td><td>0.382</td><td>0.398</td></tr><tr><td rowspan="3">Vox-Fusion[45]</td><td>PSNR↑</td><td>15.79</td><td>16.53</td><td>17.22</td><td>16.51</td></tr><tr><td>SSIM↑</td><td>0.653</td><td>0.711</td><td>0.677</td><td>0.68</td></tr><tr><td>LPIPS↓</td><td>0.514</td><td>0.423</td><td>0.459</td><td>0.465</td></tr><tr><td rowspan="3">Point-SLAM[30]</td><td>PSNR↑</td><td>13.87</td><td>17.61</td><td>18.93</td><td>16.8</td></tr><tr><td>SSIM↑</td><td>0.627</td><td>0.715</td><td>0.744</td><td>0.695</td></tr><tr><td>LPIPS↓</td><td>0.564</td><td>0.562</td><td>0.442</td><td>0.523</td></tr><tr><td rowspan="3">SplaTAM[16]</td><td>PSNR↑</td><td>22.63</td><td>24.55</td><td>22.71</td><td>23.29</td></tr><tr><td>SSIM↑</td><td>0.852</td><td>0.935</td><td>0.876</td><td>0.888</td></tr><tr><td>LPIPS↓</td><td>0.239</td><td>0.103</td><td>0.221</td><td>0.188</td></tr><tr><td rowspan="3">MonoGS[21]</td><td>PSNR↑</td><td>22.56</td><td>24.86</td><td>24.37</td><td>23.93</td></tr><tr><td>SSIM↑</td><td>0.774</td><td>0.8</td><td>0.823</td><td>0.799</td></tr><tr><td>LPIPS↓</td><td>0.247</td><td>0.211</td><td>0.21</td><td>0.223</td></tr><tr><td rowspan="3">Ours</td><td>PSNR↑</td><td>23.98</td><td>25.37</td><td>24.29</td><td>24.55</td></tr><tr><td>SSIM↑</td><td>0.833</td><td>0.851</td><td>0.881</td><td>0.855</td></tr><tr><td>LPIPS↓</td><td>0.209</td><td>0.191</td><td>0.196</td><td>0.199</td></tr></table>

Datasets To evaluate the performance of GI-SLAM, we conducted experiments on the EuRoC[3] and TUMRGBD[32] datasets. The EuRoC dataset contains stereo vision data and IMU readings, making it suitable for evaluating the model in a stereo $^ +$ IMU configuration. Although the TUM-RGBD dataset only includes accelerometer data, it is a standard benchmark in SLAM research. Therefore, we ensured compatibility with IMU data containing only accelerometer information to validate our model under monocular $+ \mathrm { I M U }$ and RGBD $^ +$ IMU settings.

Metric For camera tracking accuracy, we employed the root mean square error (RMSE) of the Absolute Trajectory Error (ATE) calculated on keyframes. To assess mapping quality, we followed the photometric rendering quality metrics used in MonoGS[21], including PSNR, SSIM, and LPIPS. These metrics were computed every five frames, excluding keyframes (training viewpoints).

Baselines We compared GI-SLAM against state-of-theart open-source methods in the 3DGS SLAM domain, including MonoGS[21] and SplaTAM[16]. Additionally, we benchmarked against earlier learning-based methods and NeRF-based approaches. For camera tracking accuracy, comparisons were made with advanced learning-based direct visual odometry (VO) methods and 3DGS SLAM methods, such as DepthCov[8], DROIDSLAM[34], SplaTAM[16] and MonoGS[21]. For mapping quality, comparisons were conducted with current leading NeRF-based SLAM methods[30, 45, 48] and 3DGS SLAM approache[16, 21].

# 4.2. Results & Discussion

Camera Tracking Results We evaluated the camera tracking performance of GI-SLAM on both the TUM and EuRoC datasets. On TUM, experiments were conducted under monocular and RGBD settings, while on EuRoC a stereo configuration was employed. As shown in Tab. 1 (TUM results for monocular and RGBD) and Tab. 5 (EuRoC stereo results), GI-SLAM consistently achieves lower RMSE values for ATE on keyframes compared to current state-of-the-art approaches. In particular, the integration of our IMU loss function plays a significant role in stabilizing pose estimates under challenging conditions. For instance, in the monocular setup our approach reduces the trajectory error over the prior SOTA baseline[21] by more than $20 \%$ from $3 . 8 8 \mathrm { c m }$ to $2 . 8 0 \mathrm { c m }$ .

The benefits of fusing IMU data become more apparent in scenarios with rapid camera motion or low-texture environments, where visual information alone might be insufficient. As illustrated in Fig. 2, our approach enables the camera tracking to achieve higher accuracy at turns. The inclusion of inertial data not only improves accuracy but also enhances the robustness of the tracking process over longer sequences, effectively mitigating drift. These improvements underscore the importance of our proposed IMU integration strategy, as reflected in the quantitative results.

Rendering Quality Results Mapping quality was assessed using photometric rendering metrics such as PSNR, SSIM, and LPIPS, computed on non-keyframe views at regular intervals. Tab. 2 summarize the performance on the TUM dataset for RGBD setup. GI-SLAM demonstrates superior rendering performance, with notable improvements in PSNR and SSIM values and a corresponding decrease in LPIPS scores compared to the baseline methods. For example, when compared to MonoGS[21], the PSNR improved from 23.93 to 24.55, with similar trends observed in SSIM and LPIPS metrics.

These gains can be largely attributed to our novel keyframe selection strategy based on motion constraint and Gaussian covisibility. By ensuring that only frames with minimal motion blur are selected for mapping, our method produces reconstructions that are both visually consistent and photorealistic. The improvements in rendering quality metrics indicate that our approach effectively reduces artifacts and preserves fine scene details, thus providing a more reliable basis for downstream tasks that depend on accurate scene representations.

Table 3. Ablation Study with/without IMU on the TUM(ATE RMSE ↓ [cm]).   

<table><tr><td>Input</td><td>IMU</td><td>fr1/desk</td><td>fr2/xyz</td><td>Avg.</td></tr><tr><td rowspan="2">Monocular</td><td>w/</td><td>1.98</td><td>3.27</td><td>2.63</td></tr><tr><td>w/o</td><td>3.51</td><td>4.29</td><td>3.90</td></tr><tr><td rowspan="2">RGBD</td><td>w/</td><td>1.34</td><td>1.26</td><td>1.30</td></tr><tr><td>w/o</td><td>1.55</td><td>1.34</td><td>1.45</td></tr></table>

Ablative Analysis To further validate the contributions of our two main innovations, we performed ablative studies by selectively disabling each component. First, when the IMU fusion module was removed, the camera tracking performance deteriorated noticeably. The ATE RMSE increased from $2 . 6 3 ~ \mathrm { c m }$ (with IMU) to $3 . 9 0 ~ \mathrm { c m }$ (without IMU) under monocular conditions, and from $1 . 3 0 \mathrm { c m }$ to $1 . 4 5 ~ \mathrm { c m }$ in RGBD configurations, as detailed in Tab. 3. This result confirms that the IMU loss function is essential for enhancing the stability and accuracy of pose estimation, particularly in dynamic or visually challenging scenarios.

In a separate experiment, we replaced our motionconstrained keyframe selection strategy with a conventional selection mechanism that does not account for motion blur. This modification led to a decline in mapping quality: PSNR decreased from 24.55 to 23.78, SSIM dropped from 0.855 to 0.818, and LPIPS increased from 0.199 to 0.211, as shown in Tab. 4. These findings emphasize that careful keyframe selection is crucial for achieving high-quality map reconstructions, as it effectively filters out frames that could introduce noise and artifacts due to motion blur.

Table 4. Ablation study of rendering on TUM with/without motion constraint as a factor in keyframe selection.   

<table><tr><td>Motion constraint</td><td>Metric</td><td>fr1/desk</td><td>fr2/xyz</td><td>fr3/office</td><td>Avg.</td></tr><tr><td rowspan="3">w/</td><td>PSNR↑</td><td>23.98</td><td>25.37</td><td>24.29</td><td>24.55</td></tr><tr><td>SSIM↑</td><td>0.833</td><td>0.851</td><td>0.881</td><td>0.855</td></tr><tr><td>LPIPS↓</td><td>0.209</td><td>0.191</td><td>0.196</td><td>0.199</td></tr><tr><td rowspan="3">w/o</td><td>PSNR↑</td><td>22.91</td><td>24.27</td><td>24.15</td><td>23.78</td></tr><tr><td>SSIM↑</td><td>0.802</td><td>0.813</td><td>0.839</td><td>0.818</td></tr><tr><td>LPIPS↓</td><td>0.225</td><td>0.197</td><td>0.212</td><td>0.211</td></tr></table>

Table 5. Camera tracking results on 3 easy sequences of EuRoC(ATE RMSE ↓ [cm]).   

<table><tr><td>Methods</td><td>mh01</td><td>mh02</td><td>v101</td><td>Avg.</td></tr><tr><td>MonoGS[21]</td><td>13.09</td><td>8.24</td><td>9.73</td><td>19.35</td></tr><tr><td>Ours</td><td>9.77</td><td>6.91</td><td>6.81</td><td>7.83</td></tr></table>

The ablative analysis highlights that both the IMU integration and the advanced keyframe selection strategy are vital components of GI-SLAM. Their combined effect not only improves camera tracking accuracy but also significantly enhances the rendering quality of the generated maps.

In summary, the experimental results demonstrate the effectiveness and robustness of GI-SLAM across various sensor configurations and datasets. The integration of IMU data via a dedicated loss function, coupled with an intelligent keyframe selection strategy, contributes to significant improvements in both tracking accuracy and mapping quality. These findings validate our design choices and suggest that GI-SLAM is well-suited for real-world 3DGS SLAM applications.

# 5. Conclusion

In this paper, we proposed GI-SLAM, a novel 3DGS SLAM framework that integrates IMU data through a dedicated loss function and employs a motion-constrained keyframe selection strategy based on Gaussian co-visibility, leading to improved camera tracking accuracy and mapping quality as demonstrated on the TUM and EuRoC datasets. However, our approach does not explicitly address the noise inherent in IMU measurements, and the metric scale ambiguity in monocular SLAM configurations remains unresolved. We plan to tackle these challenges in future work to further enhance the robustness and applicability of GI-SLAM.

# References

[1] Michael Bloesch, Tristan Laidlow, Ronald Clark, Stefan Leutenegger, and Andrew Davison. Learning meshes for

dense visual slam. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 58545863, 2019. 1 [2] Simon Boche, Xingxing Zuo, Simon Schaefer, and Stefan Leutenegger. Visual-inertial slam with tightly-coupled dropout-tolerant gps fusion. In 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 70207027, 2022. 1   
[3] Michael Burri, Janosch Nikolic, Pascal Gohl, Thomas Schneider, Joern Rehder, Sammy Omari, Markus W Achtelik, and Roland Siegwart. The euroc micro aerial vehicle datasets. Int. J. Rob. Res., 35(10):11571163, 2016. 6   
[4] Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel, and Juan D. Tardós. Orb-slam3: An accurate open-source library for visual, visualinertial, and multimap slam. IEEE Transactions on Robotics, 37(6): 18741890, 2021. 1   
[5] Chuchu Chen, Patrick Geneva, Yuxiang Peng, Woosik Lee, and Guoquan Huang. Monocular visual-inertial odometry with planar regularities. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 62246231, 2023. 1 [6] Jiawen Chen, Dennis Bautembach, and Shahram Izadi. Scalable real-time volumetric surface reconstruction. ACM Trans. Graph., 32(4), 2013. 2 [7] Angela Dai, Matthias NieBner, Michael Zollhöfer, Shahram Izadi, and Christian Theobalt. Bundlefusion: Real-time globally consistent 3d reconstruction using on-the-fly surface reintegration. ACM Trans. Graph., 36(4), 2017. 2 [8] Eric Dexheimer and Andrew J. Davison. Learning a depth covariance function. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1312213131, 2023. 6, 7   
[9] Hao Du, Peter Henry, Xiaofeng Ren, Marvin Cheng, Dan B. Goldman, Steven M. Seitz, and Dieter Fox. Interactive 3d modeling of indoor environments with a consumer depth camera. page 7584, New York, NY, USA, 2011. Association for Computing Machinery. 1, 2   
[10] Kshitij Goel and Wennie Tabib. Incremental multimodal surface mapping via self-organizing gaussian mixture models. IEEE Robotics and Automation Letters, 8(12):83588365, 2023. 2   
[11] Kshitij Goel, Nathan Michael, and Wennie Tabib. Probabilistic point cloud modeling via self-organizing gaussian mixture models. IEEE Robotics and Automation Letters, 8(5): 25262533, 2023. 2   
[12] Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Yeung. Photo-slam: Real-time simultaneous localization and photorealistic mapping for monocular, stereo, and rgb-d cameras. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2158421593, 2024. 1, 2   
[13] Jiahui Huang, Shi-Sheng Huang, Haoxuan Song, and ShiMin Hu. Di-fusion: Online implicit 3d reconstruction with deep priors. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 89288937, 2021.2   
[14] Shi-Sheng Huang, Haoxiang Chen, Jiahui Huang, Hongbo Fu, and Shi-Min Hu. Real-time globally consistent 3d reconstruction with semantic priors. IEEE Transactions on Visualization and Computer Graphics, 29(4):19771991, 2023. 2   
[15] Mohammad Mahdi Johari, Camilla Carta, and François Fleuret. Eslam: Efficient dense slam system based on hybrid representation of signed distance fields. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1740817419, 2023. 1, 2   
[16] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat, track & map 3d gaussians for dense rgb-d slam. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2135721366, 2024. 1, 2, 6, 7   
[17] Maik Keller, Damien Lefloch, Martin Lambers, Shahram Izadi, Tim Weyrich, and Andreas Kolb. Real-time 3d reconstruction in dynamic scenes using point-based fusion. In 2013 International Conference on 3D Vision - 3DV 2013, pages 18, 2013. 2   
[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4), 2023. 1,2   
[19] Christian Kerl, Jürgen Sturm, and Daniel Cremers. Dense visual slam for rgb-d cameras. In 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 21002106, 2013. 2   
[20] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian representation for radiance field. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21719 21728, 2024. 3   
[21] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and Andrew J. Davison. Gaussian splatting slam. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1803918048, 2024. 1, 2, 4, 5, 6, 7, 8   
[22] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: representing scenes as neural radiance fields for view synthesis. Commun. ACM, 65(1):99106, 2021. 1, 2   
[23] Manasi Muglikar, Zichao Zhang, and Davide Scaramuzza. Voxel map for visual slam. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 4181 4187, 2020. 1   
[24] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 41(4), 2022. 1   
[25] Raúl Mur-Artal and Juan D. Tardós. Orb-slam2: An opensource slam system for monocular, stereo, and rgb-d cameras. IEEE Transactions on Robotics, 33(5):12551262, 2017. 1   
[26] Richard A. Newcombe, Shahram Izadi, Otmar Hilliges, David Molyneaux, David Kim, Andrew J. Davison, Pushmeet Kohi, Jamie Shotton, Steve Hodges, and Andrew Fitzgibbon. Kinectfusion: Real-time dense surface mapping and tracking. In 2011 10th IEEE International Symposium on Mixed and Augmented Reality, pages 127136, 2011. 2   
[27] Matthias NieBner, Michael Zollhöfer, Shahram Izadi, and Marc Stamminger. Real-time 3d reconstruction at scale using voxel hashing. ACM Trans. Graph., 32(6), 2013. 2   
[28] Helen Oleynikova, Zachary Taylor, Marius Fehr, Roland Siegwart, and Juan Nieto. Voxblox: Incremental 3d euclidean signed distance fields for on-board mav planning. In 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 13661373, 2017. 2   
[29] Tong Qin, Peiliang Li, and Shaojie Shen. Vins-mono: A robust and versatile monocular visual-inertial state estimator. IEEE Transactions on Robotics, 34(4):10041020, 2018. 1   
[30] Erik Sandström, Yue Li, Luc Van Gool, and Martin R. Oswald. Point-slam: Dense neural point cloud-based slam. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 1838718398, 2023. 2, 6, 7   
[31] Thomas Schöps, Torsten Sattler, and Marc Pollefeys. Bad slam: Bundle adjusted direct rgb-d slam. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 134144, 2019. 2   
[32] Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers. A benchmark for the evaluation of rgb-d slam systems. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 573580, 2012. 6   
[33] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J. Davison. imap: Implicit mapping and positioning in real-time. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 62096218, 2021. 1, 2   
[34] Zachary Teed and Jia Deng. Droid-slam: deep visual slam for monocular, stereo, and rgb-d cameras. In Proceedings of the 35th International Conference on Neural Information Processing Systems, Red Hook, NY, USA, 2021. Curran Associates Inc. 6, 7   
[35] Andreas L. Teigen, Yeonsoo Park, Annette Stahl, and Rudolf Mester. Rgb-d mapping and tracking in a plenoxel radiance field. In 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 33303339, 2024. 2   
[36] Fabio Tosi, Youmin Zhang, Ziren Gong, Erik Sandström, Stefano Mattoccia, Martin R Oswald, and Matteo Poggi. How nerfs and 3d gaussian splatting are reshaping slam: a survey. arXiv preprint arXiv:2402.13255, 4, 2024. 2   
[37] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Coslam: Joint coordinate and sparse parametric encodings for neural real-time slam. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1329313302, 2023. 1, 2   
[38] Kaixuan Wang, Fei Gao, and Shaojie Shen. Real-time scalable dense surfel mapping. In 2019 International Conference on Robotics and Automation (ICRA), pages 69196925, 2019. 1, 2   
[39] Weihan Wang, Jiani Li, Yuhang Ming, and Philippos Mordohai. Edi: Eskf-based disjoint initialization for visual-inertial slam systems. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 14661472, 2023. 1   
[40] Silvan Weder, Johannes Schönberger, Marc Pollefeys, and Martin R. Oswald. Routedfusion: Learning real-time depth map fusion. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 48864896, 2020. 2   
[41] Jiaxin Wei and Stefan Leutenegger. Gsfusion: Online rgb-d mapping where gaussian splatting meets tsdf fusion. IEEE Robotics and Automation Letters, 9(12):1186511872, 2024. 1   
[42] Thomas Whelan, Stefan Leutenegger, Renato Salas Moreno, Ben Glocker, and Andrew Davison. Elasticfusion: Dense slam without a pose graph. In Proceedings of Robotics: Science and Systems, Rome, Italy, 2015. 2   
[43] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d gaussian splatting. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19595 19604, 2024. 1, 3   
[44] Guandao Yang, Xun Huang, Zekun Hao, Ming-Yu Liu, Serge Belongie, and Bharath Hariharan. Pointflow: 3d point cloud generation with continuous normalizing flows. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 45404549, 2019. 1   
[45] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian Liu, and Guofeng Zhang. Vox-fusion: Dense tracking and mapping with voxel-based neural implicit representation. In 2022 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), pages 499507, 2022. 1, 2, 6, 7   
[46] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R. Oswald. Gaussian-slam: Photo-realistic dense slam with gaussian splatting, 2023. 3   
[47] Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo Poggi. Go-slam: Global optimization for consistent 3d instant reconstruction. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 37043714, 2023. 2   
[48] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc Pollefeys. Nice-slam: Neural implicit scalable encoding for slam. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1277612786, 2022. 1, 2, 6,7