# 从roadSide.cpp发送出去的消息，主要存储点云配准的结果
Header header
int32 keyFrameId     # 关键帧ID
float64 timestamp      # 从VehicleInfoHandler获取的时间戳,对应关键帧的时间戳
geometry_msgs/Pose pose # 路侧lidar和车载lidar的位姿，vehicle frame to roadSide frame