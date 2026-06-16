# 问题

1.当前版本针对于`Azure Kinect采集的数据`因为`rgb`和`depth`的分辨率不同而设计的`resize`功能和`k4a_transformation functions of SDK`中的`k4a_transformation_depth_image_to_color_camera`和`k4a_transformation_color_image_to_depth_camera`功能，二者适配图片分辨率的原理是否相同（裁剪or其他）

2.[badslam](https://github.com/ETH3D/badslam)是否解决了问题1。需要研究其代码，尤其是'https://github.com/ETH3D/badslam/blob/master/applications/badslam/src/badslam/input_azurekinect.cc'

3.GI-SLAM的IMU损失函数策略

4.Photo-SLAM的高斯金字塔策略

5.Splatam的各向同性高斯策略