#define PCL_NO_PRECOMPILE
#include "utility.h"
#include "lio_sam_6axis/cloud_info.h"
#include "lio_sam_6axis/road_registration.h"
#include "pointRegistration.h"
#include <ros/time.h>
#include <tf2_eigen/tf2_eigen.h> // for Eigen与geometry_msgs的转换
#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LocalCartesian.hpp>
#include <GeographicLib/Geoid.hpp>

struct VelodynePointXYZIRT {
    PCL_ADD_POINT4D

    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
                                   (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
                                           (uint16_t, ring, ring)(float, time, time)
)

struct Velodyne_M1600PointXYZIRT {
   PCL_ADD_POINT4D;
  uint8_t intensity;
  uint8_t ring;
  uint32_t timestampSec;
  uint32_t timestampNsec;
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(
    Velodyne_M1600PointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(uint8_t, intensity, intensity)(
        uint8_t, ring, ring)(uint32_t, timestampSec, timestampSec)(uint32_t, timestampNsec, timestampNsec))

struct PointXYZIRPYT {
    PCL_ADD_POINT4D

    PCL_ADD_INTENSITY;  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
} EIGEN_ALIGN16;  // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(
        PointXYZIRPYT,
        (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
                float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time,
                                                                         time))
typedef PointXYZIRPYT PointTypePose;

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

// 通用模板函数（支持所有符合字段要求的点类型）
template <typename PointT>
std::vector<Eigen::Vector4f> ConvertCloudToEigenVector4f(
    const typename pcl::PointCloud<PointT>::Ptr& cloud,
    bool useIntensity = true) 
{
    std::vector<Eigen::Vector4f> eigenPoints;
    if (!cloud || cloud->empty()) {
        ROS_WARN("Input cloud is empty!");  // 根据实际日志库调整
        return eigenPoints;
    }

    eigenPoints.reserve(cloud->size());
    for (const auto& point : *cloud) {
        Eigen::Vector4f vec;
        vec[0] = point.x;
        vec[1] = point.y;
        vec[2] = point.z;
        vec[3] = useIntensity ? point.intensity : 1.0f; // 使用强度或填充1.0
        eigenPoints.emplace_back(vec);
    }

    return eigenPoints;
}

class RoadSide : public ParamServer {
private:
    // 静态成员声明（类内）
    static std::once_flag ringCheckFlag, deskewCheckFlag;
    static bool hasRing;
    static bool hasTimestamp;
    GeographicLib::LocalCartesian geoConverter;

    ros::Subscriber subRoadSideCloud;
    ros::Subscriber subGNSSOrigin;
    ros::Subscriber subVehicleCloud; // 订阅后端发送来的关键帧的信息
    ros::Publisher pubRoadSideRegister; // 发布路侧配准的结果
    bool isWithinRoadRange; // 车子是否在路侧范围内
    Eigen::Matrix4d T_GNSSN_Road; // 路侧lidar到gnss的n系的4乘4的变换矩阵

    // 路侧lidar信息
    std::deque<sensor_msgs::PointCloud2> roadCloudQueue;
    std_msgs::Header roadCloudHeader; // 当前路侧lidar的点云消息头
    double timeRoadKeyFrame; // 和关键帧匹配成功的路侧lidar的时间戳
    double timeRoadCurrent; // 路侧点云回调函数进来消息的时间戳
    double timeRoadKeyFrameEnd; // 当前路侧lidar的结束时间戳
    int deskewFlag; // 畸变校正的标志
    std::mutex mtxRoadCloud;
    double roadCloudTime;
    PointType pose3DRoadSide; // 以gnss的n系为原点的路侧lidar的平移量，也就是路侧lidar在gnss的n系下的位置

    // 车载传感器的关键帧信息
    lio_sam_6axis::cloud_info vehicleOptmCloudInfo; // 关键帧的信息，下面的变量都是从这个变量中提取的
    ros::Time timeOptmizationStamp; // 当前关键帧的时间戳，ros::Time类型
    double timeOptmizationCur; // 当前关键帧的时间戳，double类型
    pcl::PointCloud<PointType>::Ptr vehicleCloudRawDS; // 当前关键帧经过去畸变的点云，同时也进行了降采样
    pcl::PointCloud<PointType>::Ptr vehicleCloudCornerSurfDS; // 当前关键帧的角点和面点，同时也进行了降采样
    pcl::PointCloud<PointTypePose>::Ptr keyFramePoses6D; // 所有的关键帧6D位姿，里面的光强参数代表索引
    PointType latestKeyPoint; // 最新时刻的关键帧位置，光强代表关键帧的索引

    // 点云配准的信息
    std::deque<lio_sam_6axis::road_registration> roadRegistrationDeque;
    std::mutex registDequeMutex; // 互斥锁
    const size_t maxDequeSize = 400;  // 队列最大长度
    Eigen::Vector3d GNSSOriginLLA; // 原点的纬度、经度和高度


public:
    RoadSide() : deskewFlag(0), isWithinRoadRange(false) {
        // 订阅后端已经优化后的点云信息和优化后的位姿信息，以及关键帧的id
        subVehicleCloud = nh.subscribe<lio_sam_6axis::cloud_info>("lio_sam_6axis/mapping/slam_info", 1000, &RoadSide::VehicleInfoHandler, this,
                ros::TransportHints().tcpNoDelay());
        // 订阅路侧点云话题
        subRoadSideCloud = nh.subscribe<sensor_msgs::PointCloud2>(roadSidePointTopic, 10, &RoadSide::RoadSideCloudHandler, this,
                ros::TransportHints().tcpNoDelay());
        subGNSSOrigin = nh.subscribe<geometry_msgs::Vector3>("lio_sam_6axis/mapping/originLLA", 30, &RoadSide::GNSSOriginHandler, this,
            ros::TransportHints().tcpNoDelay());
        pubRoadSideRegister = nh.advertise<lio_sam_6axis::road_registration>("lio_sam_6axis/roadside/registration", 1);
        
        allocateMemory();
        resetParameters(true, true); // 清理路侧和车载信息
    }


    void allocateMemory() {
        // 路侧信息的初始化

        // 车载信息的初始化
        vehicleCloudRawDS.reset(new pcl::PointCloud<PointType>());
        vehicleCloudCornerSurfDS.reset(new pcl::PointCloud<PointType>());
        keyFramePoses6D.reset(new pcl::PointCloud<PointTypePose>());
    }

    void resetParameters(bool clearRoadside, bool clearVehicle) {
        if (clearRoadside) {
            // 慢慢清空路侧信息，roadCloudQueue是一个双端队列
            std::lock_guard<std::mutex> lock(mtxRoadCloud);
            // 当队列大小超过 maxDequeSize 时，删除多余的旧数据
            while (roadCloudQueue.size() > maxDequeSize) {
                roadCloudQueue.pop_front();
            }
        }
        if (clearVehicle) {
            // 一次性清空车载信息
            vehicleCloudRawDS->clear();
            vehicleCloudCornerSurfDS->clear();
            keyFramePoses6D->clear();
        }
    }

    ~RoadSide() {}

    void GNSSOriginHandler(const geometry_msgs::Vector3ConstPtr &msgIn) {
        // 只接收一次：接收到数据后赋值并打印
        GNSSOriginLLA[0] = msgIn->x;
        GNSSOriginLLA[1] = msgIn->y;
        GNSSOriginLLA[2] = msgIn->z;
        ROS_INFO_STREAM("[updateInitialGuess] originLLA: " << std::fixed << std::setprecision(9) << GNSSOriginLLA.transpose());                    /** set your map origin points */
        // 取消订阅，确保以后不再执行该回调
        subGNSSOrigin.shutdown();
    }

    // 路侧点云的回调函数
    void RoadSideCloudHandler(const sensor_msgs::PointCloud2ConstPtr &roadCloudMsg) {
        timeRoadCurrent = roadCloudMsg->header.stamp.toSec(); // 记录当前回调时间，与点云配准后的关键帧时间戳对比
        std::unique_lock<std::mutex> lock(mtxRoadCloud);
        roadCloudQueue.push_back(*roadCloudMsg);
        if (roadCloudQueue.size() <= 2) {
            return;
        }
        if (isWithinRoadRange) {
            // 如果车子在路侧范围内，则保留最大长度的数据
            while (roadCloudQueue.size() > maxDequeSize) {
                roadCloudQueue.pop_front();
            }
        } else {
            // 如果车子不在范围内，则只保留最大长度1/3的数据
            while (roadCloudQueue.size() > maxDequeSize / 3) {
                roadCloudQueue.pop_front();
            }
        }
    }

    bool ChangeNFrame() {
        if (GNSSOriginLLA.isZero(1e-4)) {
            // 打印地图原点（LLA）的坐标
            return false;
        }

        /** set ENU坐标系 origin points */
        geoConverter.Reset(GNSSOriginLLA[0], GNSSOriginLLA[1], GNSSOriginLLA[2]);
        // WGS84->ENU, 路侧lidar对应的n系，这个n系对应的纬度，经度，高度
        Eigen::Vector3d enu; // enu为路侧lidar对应的n系原点在GNSS里程计n系下的坐标
        geoConverter.Forward(30.3994217, 114.1404822, 0, enu[0], enu[1], enu[2]);  // 里面的变量为路侧lidar原点的经纬度和高度
        ROS_INFO_STREAM("ENU coordinates: [" << enu[0] << ", " << enu[1] << ", " << enu[2] << "]");

        // 构造T_GNSSN_HDMapN：旋转为单位阵，平移量为enu的三个分量
        Eigen::Matrix4d T_GNSSN_HDMapN = Eigen::Matrix4d::Identity();
        T_GNSSN_HDMapN(0,3) = enu[0];
        T_GNSSN_HDMapN(1,3) = enu[1];
        T_GNSSN_HDMapN(2,3) = enu[2];

        T_GNSSN_Road = T_GNSSN_HDMapN * T_HDMapN_Road;
        Eigen::Vector3d T_GNSSN_Road_Trans = T_GNSSN_Road.block<3, 1>(0, 3);
        pose3DRoadSide.getVector3fMap() = T_GNSSN_Road_Trans.cast<float>();
        return true;
    }

    void VehicleInfoHandler(const lio_sam_6axis::cloud_infoConstPtr &msgIn) {

        // 更改路侧lidar的n系的原点位置，因为gnss的n系位置是第一个时刻的gnss的位置对应的n系
        static bool hasChangedNframe = false;
        // 如果未成功，则每次尝试调用 ChangeNFrame
        if (!hasChangedNframe) {
            hasChangedNframe = ChangeNFrame();
            if (!hasChangedNframe) {
                if (debugRoadSide) {
                    ROS_INFO_STREAM("ChangeNFrame failed. Will try again next time.");
                }
                return;
            }
        }
        
        static int addRoadNum = 0;
        ros::WallTime procStart = ros::WallTime::now();
        timeOptmizationStamp = msgIn->header.stamp;        // extract time stamp
        timeOptmizationCur = msgIn->header.stamp.toSec();

        // 获取最新时刻的关键帧位姿，判断当前位置是否需要和路侧lidar进行点云配准
        pcl::fromROSMsg(msgIn->key_frame_poses, *keyFramePoses6D);  // keyFramePoses6D存储所有的关键帧位姿
        updateLatestKeyPoint();  // 更新最新关键帧的位置latestKeyPoint，同时返回车子的4乘4位姿矩阵
        Eigen::Matrix4d initT_GNSSN_Vehicle = GetVehiclePose6D(keyFramePoses6D->back());  // 获取车子的6D位姿，然后通过四元数转成4乘4的矩阵

        double keyFrameID = keyFramePoses6D->back().intensity;  // 关键帧的索引
        double keyFrameTime = keyFramePoses6D->back().time;  // 关键帧的时间戳
        // 判断下timeOptmizationCur和keyFrameTime是否相等
        if (fabs(timeOptmizationCur - keyFrameTime) > 0.0005) {
            ROS_WARN("Time stamp in cloud_info and cloud_deskewed is not synchronized!");
            return;
        }

        // 判断车子是否在路侧lidar范围内
        float rangeRoadVehicle = pointDistance(pose3DRoadSide, latestKeyPoint);
        // 在VehicleInfoHandler中调用updateLatestKeyPoint后，插入如下日志输出：
        isWithinRoadRange = (rangeRoadVehicle <= rangeRoadVehicleThr) ? true : false;
        if (!isWithinRoadRange) {
            if (debugRoadSide) {
                ROS_INFO_STREAM("roadSide.cpp Vehicle distance from roadside lidar: " << rangeRoadVehicle << " m");
            }
            resetParameters(false, true);  // 清空车载信息
            return;
        }

        ROS_INFO("------------------------------------------------");
        if (debugRoadSide) {
            ROS_INFO_STREAM("[KeyFrame]: keyFrameTime = " << std::fixed << std::setprecision(4) 
                             << keyFrameTime << " s, keyFrameID = " << keyFrameID);
        }
        // 找到在关键帧时间戳附近的路侧点云
        pcl::PointCloud<PointXYZIRT>::Ptr curSyncedRoadCloud = syncRoadSideCloud(timeOptmizationCur, toleranceTime);
        if(curSyncedRoadCloud == nullptr) {
            ROS_INFO("Warning! In the roadside lidar range, But there is no suitable roadside point cloud!");
            return;
        }

        // 提取关键帧的信息，同时转换路侧和车载点云到Eigen格式
        vehicleOptmCloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_deskewed, *vehicleCloudRawDS);  // 当前关键帧经过去畸变的点云
        pcl::fromROSMsg(msgIn->key_frame_cloud, *vehicleCloudCornerSurfDS);  // 当前关键帧的角点和面点组成的点云
        // 保存点云部分（在 debugRoadSide 为 true 时执行）
        if (debugRoadSide) {
            ROS_INFO_STREAM("[Point] Deskewed cloud size: " << vehicleCloudRawDS->points.size() 
                                << " | Corner & surf point cloud size: " << vehicleCloudCornerSurfDS->points.size());
            // 保存车载点云，文件名格式：<keyFrameTime>_VehicleleCloud.pcd
            std::ostringstream ss;
            ss << "/home/zhao/Data/tunnelData/data_2025220163953/PCD/"
            << std::fixed << std::setprecision(4) << keyFrameTime << "_VehicleleCloud.pcd";
            std::string filename = ss.str();
            pcl::io::savePCDFileBinary(filename, *vehicleCloudRawDS);
            ROS_INFO_STREAM("Saved vehicleCloudRawDS to " << filename << " with "
                                << vehicleCloudRawDS->points.size() << " points.");

            // 保存路侧点云，文件名格式：<roadCloudTime>_roadSide.pcd
            std::ostringstream ss2;
            ss2 << "/home/zhao/Data/tunnelData/data_2025220163953/PCD/"
                << std::fixed << std::setprecision(4) << roadCloudTime << "_roadSide.pcd";
            std::string filename2 = ss2.str();
            pcl::io::savePCDFileBinary(filename2, *curSyncedRoadCloud);
            ROS_INFO_STREAM("Saved roadSide point cloud to " << filename2 << " with " 
                                << curSyncedRoadCloud->points.size() << " points.");
        }
        auto curRoadEigenPoints = ConvertCloudToEigenVector4f<PointXYZIRT>(curSyncedRoadCloud); // 目标点云
        auto vehicleEigenPoints = ConvertCloudToEigenVector4f<PointType>(vehicleCloudCornerSurfDS); // 源点云

        if (debugRoadSide) {
            std::ostringstream oss;
            oss << "[EigenClouds] Roadside size: " << curRoadEigenPoints.size()
                << " | Vehicle size: " << vehicleEigenPoints.size();
            ROS_INFO_STREAM(oss.str());
        }
        
        Eigen::Matrix4d T_Road_GNSSN = T_GNSSN_Road.inverse();  // n frame to roadLidar frame
        Eigen::Matrix4d initT_Road_Vehicle = T_Road_GNSSN * initT_GNSSN_Vehicle;  // vehicle frame to roadLidar frame 初始变换矩阵
        // 调用配准函数
        Eigen::Matrix4d T_Road_Vehicle = CloudRegistrationGicp(curRoadEigenPoints, vehicleEigenPoints, initT_Road_Vehicle, roadCloudDs);
        if (!T_Road_Vehicle.allFinite()) {
            std::cerr << "CloudRegistration has WRONG result!" << std::endl;
        }
        Eigen::Matrix4d T_GNSSN_Vehicle = T_GNSSN_Road * T_Road_Vehicle;
        StoreRegistrationResult(keyFrameID, timeOptmizationCur, T_GNSSN_Vehicle);  // 存储配准结果
        PublishRegistration();

        if (debugRoadSide) {
            ros::WallDuration procTime = ros::WallTime::now() - procStart;
            double deltaCurrToOpt = timeRoadCurrent - timeOptmizationCur;
            ROS_INFO_STREAM("[Time] RoadCurrent=" << std::fixed << std::setprecision(4) << timeRoadCurrent << " s | "
                << "OptCur=" << std::fixed << std::setprecision(4) << timeOptmizationCur << " s | "
                << "Delta=" << std::fixed << std::setprecision(6) << abs(timeRoadCurrent - timeOptmizationCur) << " s | "
                << "Processing=" << std::fixed << std::setprecision(6) << procTime.toSec() << " sec");
        }
    }

    void StoreRegistrationResult(double keyFrameId, double timestamp, const Eigen::Matrix4d& transform) {
        lio_sam_6axis::road_registration msg;
        // 修正时间戳设置
        msg.header.stamp = ros::Time().fromSec(timestamp);
        msg.header.frame_id = "road_side";
        
        // 修正类型转换
        msg.keyFrameId = static_cast<int32_t>(keyFrameId);
        msg.timestamp = timestamp;
        
        // 修正Eigen转Pose
        const Eigen::Isometry3d transform_iso = Eigen::Isometry3d(transform);
        msg.pose = tf2::toMsg(transform_iso);  // 使用tf2的转换方法
        
        // 线程安全写入
        {
            std::lock_guard<std::mutex> lock(registDequeMutex);
            roadRegistrationDeque.push_back(msg);
        }
    }

    // 发布路侧点云和车载点云的配准结果
    void PublishRegistration() {
        std::lock_guard<std::mutex> lock(registDequeMutex);
        if (roadRegistrationDeque.empty()) {
            return;
        }

        // 发布最旧数据
        const auto& msg = roadRegistrationDeque.front();
        pubRoadSideRegister.publish(msg);
        roadRegistrationDeque.pop_front();
        ROS_INFO_STREAM("[Publish] registration: FrameID=" << msg.keyFrameId 
                            << " Time=" << std::fixed << msg.timestamp);
    }

    pcl::PointCloud<PointXYZIRT>::Ptr syncRoadSideCloud(double keyframeTimestamp, double timeTol) {
        sensor_msgs::PointCloud2 syncedCloudMsg;
        bool hasSynced = false;

        { // 锁保护队列操作
            std::unique_lock<std::mutex> lock(mtxRoadCloud);
            while (!roadCloudQueue.empty()) {
                const auto& frontCloud = roadCloudQueue.front();
                roadCloudTime = frontCloud.header.stamp.toSec();

                if (roadCloudTime < keyframeTimestamp - timeTol) {
                    roadCloudQueue.pop_front();
                } else if (roadCloudTime > keyframeTimestamp + timeTol) {
                    break;
                } else {
                    syncedCloudMsg = std::move(frontCloud); // 移动语义减少拷贝
                    roadCloudQueue.pop_front();
                    hasSynced = true;
                    if (debugRoadSide) {
                        ROS_INFO_STREAM("[Sync Time met]: roadCloudTime = " << std::fixed << std::setprecision(4) 
                                         << roadCloudTime << " s, keyframeTimestamp = " << keyframeTimestamp << " s");
                    }

                    break;
                }
            }
            if (debugRoadSide) {
                // 打印队列数量
                ROS_INFO_STREAM("[RoadCloudQueue size]: " << roadCloudQueue.size() << " | hasSynced: " << hasSynced 
                                << " | keyframeTimestamp: " << std::fixed << std::setprecision(5) << keyframeTimestamp);
                std::ostringstream oss;
                oss << "[RoadCloudQueue timestamps]: ";
                // 打印队头三个话题时间戳
                int headCount = 0;
                for (auto it = roadCloudQueue.begin(); it != roadCloudQueue.end() && headCount < 2; ++it, ++headCount) {
                    oss << "[head " << headCount << "]: " 
                        << std::fixed << std::setprecision(6) << it->header.stamp.toSec();
                    if (headCount < 2) {
                        oss << " | ";
                    }
                }
                oss << " || "; // 分隔线
                // 打印队尾三个话题时间戳
                int tailCount = 0;
                for (auto it = roadCloudQueue.rbegin(); it != roadCloudQueue.rend() && tailCount < 2; ++it, ++tailCount) {
                    oss << "[tail " << tailCount << "]: " 
                        << std::fixed << std::setprecision(6) << it->header.stamp.toSec();
                    if (tailCount < 2) {
                        oss << " | ";
                    }
                }
                ROS_INFO_STREAM(oss.str());
            }

        }

        if (hasSynced) {
            return processRoadCloud(syncedCloudMsg); // 返回处理后的点云指针
        } else {
            return nullptr; // 未同步到数据时返回空指针
        }
    }

    pcl::PointCloud<PointXYZIRT>::Ptr processRoadCloud(sensor_msgs::PointCloud2 syncedCloudMsg) {
        // 1. 创建局部PCL点云对象（避免全局变量冲突）
        pcl::PointCloud<PointXYZIRT>::Ptr curRoadCloudPcl(new pcl::PointCloud<PointXYZIRT>);

        // 2. 检查传感器类型
        if (roadSensor != SensorType::VELODYNE && roadSensor != SensorType::LIVOX) {
            ROS_ERROR_STREAM("Unknown roadSensor type: " << int(roadSensor));
            return nullptr;
        }

        // 3. 转换ROS消息到PCL格式
        pcl::moveFromROSMsg(syncedCloudMsg, *curRoadCloudPcl);

        // 去畸变的时间戳，暂时没有用到
        roadCloudHeader = syncedCloudMsg.header;
        timeRoadKeyFrame = roadCloudHeader.stamp.toSec();
        timeRoadKeyFrameEnd = timeRoadKeyFrame + curRoadCloudPcl->points.back().time;

        if (debugLidarTimestamp) {
            std::cout << std::fixed << std::setprecision(12)
                      << "end time from pcd and size: " << curRoadCloudPcl->points.back().time
                      << ", " << curRoadCloudPcl->points.size() << std::endl;
        }
        // 4. 去NaN点
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*curRoadCloudPcl, *curRoadCloudPcl, indices);
        if (!curRoadCloudPcl->is_dense) {
            ROS_ERROR("Roadside Point cloud is not dense.");
            return nullptr;
        }

        // 5. 检查 ring 通道（仅执行一次）
        std::call_once(ringCheckFlag, [&]() {
            for (const auto& field : syncedCloudMsg.fields) {
                if (field.name == "ring") {
                    hasRing = true;
                    break;
                }
            }

            // 非Velodyne传感器必须包含 ring 通道
            if (!hasRing && roadSensor != SensorType::VELODYNE) {
                ROS_ERROR("Roadside Point cloud ring channel not available!");
            }

            // Velodyne允许无 ring 通道（仅警告）
            if (roadSensor == SensorType::VELODYNE && !hasRing) {
                ROS_WARN("Velodyne point cloud lacks 'ring' channel.");
            }
        });

        // 若非Velodyne且无 ring 通道，返回错误
        if (roadSensor != SensorType::VELODYNE && !hasRing) {
            return nullptr;
        }

        // 6. 检查时间戳字段（仅执行一次）
        std::call_once(deskewCheckFlag, [&]() {
            for (const auto& field : syncedCloudMsg.fields) {
                if (field.name == "time" || field.name == "t" || 
                    field.name == "timestamp" || field.name == "timestampSec") {
                    hasTimestamp = true;
                    break;
                }
            }

            if (!hasTimestamp) {
                ROS_WARN("Point cloud timestamp not available, deskew disabled.");
            }
        });

        // 若无时间戳字段，返回错误（根据需求可调整）
        if (!hasTimestamp) {
            return nullptr;
        }

        return curRoadCloudPcl;
    }

     // 更新最新关键帧的位置latestKeyPoint，同时返回车子的4乘4位姿矩阵
    void updateLatestKeyPoint() {
        if (!keyFramePoses6D->empty()) {
            latestKeyPoint.x = keyFramePoses6D->back().x;
            latestKeyPoint.y = keyFramePoses6D->back().y;
            latestKeyPoint.z = keyFramePoses6D->back().z;
            latestKeyPoint.intensity = keyFramePoses6D->back().intensity; // 这个是关键帧的索引
        }
    }

    Eigen::Matrix4d GetVehiclePose6D(const PointTypePose &pose) {
        // 赋值顺序已经校验过
        float pose6D[6] = {pose.roll, pose.pitch, pose.yaw, pose.x, pose.y, pose.z};
        nav_msgs::Odometry vehicleOdo;
        transformEiegn2Odom(pose.time, vehicleOdo, pose6D);

        Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
        transformation(0,3) = vehicleOdo.pose.pose.position.x;
        transformation(1,3) = vehicleOdo.pose.pose.position.y;
        transformation(2,3) = vehicleOdo.pose.pose.position.z;
        // 提取姿态信息（四元数转旋转矩阵）
        Eigen::Quaterniond q(vehicleOdo.pose.pose.orientation.w,
                               vehicleOdo.pose.pose.orientation.x,
                               vehicleOdo.pose.pose.orientation.y,
                               vehicleOdo.pose.pose.orientation.z);
        transformation.block<3,3>(0,0) = q.toRotationMatrix();
        return transformation;
    }
};

// 静态成员定义，类外，类外定义并初始化静态成员
std::once_flag RoadSide::ringCheckFlag;
std::once_flag RoadSide::deskewCheckFlag;
bool RoadSide::hasRing = false;
bool RoadSide::hasTimestamp = false;

int main(int argc, char **argv) {
    ros::init(argc, argv, "lio_sam_6axis");

    RoadSide RS;

    ROS_INFO("\033[1;32m----> RoadSide LiDAR Started.\033[0m");

    ros::MultiThreadedSpinner spinner(36);
    spinner.spin();

    return 0;
}
