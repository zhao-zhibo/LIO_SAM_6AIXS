#define PCL_NO_PRECOMPILE
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/dataset.h>  // gtsam
#include <std_srvs/Empty.h>
#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LocalCartesian.hpp>
#include <GeographicLib/Geoid.hpp>
#include <csignal>
#include <pcl/search/impl/search.hpp>

#include "dataSaver.h"
#include "lio_sam_6axis/cloud_info.h"
#include "lio_sam_6axis/road_registration.h"
#include "lio_sam_6axis/save_map.h"
#include "lio_sam_6axis/transformInterp.h"
#include "utility.h"
#include "std_msgs/Float32MultiArray.h"
#include <iostream>
#include <sstream>
#include <cstdio>
#include <string>
#include <ros/ros.h>
#include <ros/package.h>
#include <cmath> // 用于角度转换


using namespace gtsam;

using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G;  // GPS pose
using symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)
/*
 * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is
 * time stamp)
 */
class mapOptimization : public ParamServer {
public:
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;
    ros::Publisher pubGPSOdometry;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubCloudRaw;
    ros::Publisher pubLoopConstraintEdge;
    ros::Publisher pubGpsConstraintEdge;

    ros::Publisher pubSLAMInfo;
    ros::Publisher pubHessianMatrix; // 发布开始优化时的第一次的Hessian矩阵
    ros::Publisher pubGNSSOriginLLA; // 发布GNSS原点的经纬度信息给路侧节点

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subRoadSide;
    ros::Subscriber subLoop;

    ros::ServiceServer srvSaveMap;
    ros::ServiceClient srvTransformClient;  // [!++ 新增成员变量]
    std::atomic<bool> serviceAvailable{false};  // [!++ 小驼峰: serviceAvailable]
    std::thread serviceCheckThread;          // [!++ 小驼峰: serviceCheckThread]

    std::deque<nav_msgs::Odometry> gpsQueue;
    std::deque<lio_sam_6axis::road_registration> roadRegistrationQueue; // 存储从路侧lidar发回来的关键帧点云配准信息
    lio_sam_6axis::cloud_info cloudInfo; // 从laserCloudInfoHandler进行赋值，接入的话题是lio_sam_6axis/feature/cloud_info。

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> laserCloudRawKeyFrames; // 原始的经过去畸变的点云，但是还在雷达坐标系中

    std::vector<Eigen::Matrix4d> keyframePosestrans;
    std::vector<nav_msgs::Odometry> keyframeRawOdom;
    std::vector<double> keyframeTimes;
    //    std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
    std::vector<sensor_msgs::PointCloud2> keyframeCloudDeskewed;
    std::vector<double> keyframeDistances;
    std::vector<gtsam::GPSFactor> keyframeGPSfactor;

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointType>::Ptr cloudKeyGPSPoses3D;
    pcl::PointCloud<PointType>::Ptr cloudKeyroadSidePoses3D; // 存储从路侧点云配准信息中提取的关键帧位姿
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D; // 所有的关键帧6D位姿，里面的光强参数代表索引
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses2D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr
            laserCloudCornerLast;  // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr
            laserCloudSurfLast;  // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr
            laserCloudCornerLastDS;  // downsampled corner feature set from
    // odoOptimization
    pcl::PointCloud<PointType>::Ptr
            laserCloudSurfLastDS;  // downsampled surf feature set from
    // odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudRaw;    // giseop
    pcl::PointCloud<PointType>::Ptr laserCloudRawDS;  // giseop

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType>
            laserCloudOriCornerVec;  // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType>
            laserCloudOriSurfVec;  // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>>
            laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType>
            downSizeFilterSurroundingKeyPoses;        // for surrounding key poses of
    // scan-to-map optimization
    pcl::VoxelGrid<PointType> downSizeFilterRaw;

    std::unique_ptr<DataSaver> dataSaverPtr;

    bool flg_exit = false;
    int lastLoopIndex = -1;

    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur; // 当前处理帧的时间戳
    // double timeStampInitial;

    float transformTobeMapped[6]; // 0  1  2 : roll  pitch  yaw,   3 4  5: x  y  z

    std::mutex mtx;
    std::mutex mtxLoopInfo;
    std::mutex mtxGpsInfo;
    std::mutex mtxRoadSideInfo;
    std::mutex mtxGraph;
    std::mutex debugKeyFrameMutex;

    // Eigen::Affine3f transGPS;
    // Eigen::Vector3d transLLA;
    Eigen::Vector3d originLLA; // GNSS的原点的位置
    // bool gpsAvialble = false;
    bool systemInitialized = false;
    bool gpsTransfromInit = false;
    GeographicLib::LocalCartesian geo_converter;

    bool isDegenerate = false;
    cv::Mat matP;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer;  // from new to old
    map<int, int> gpsIndexContainer;   // from new to old
    map<int, int> roadIndexContainer;   // 记录因子索引 (关键帧ID -> roadsidePoses索引)
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec;
    nav_msgs::Path globalPath;
    // nav_msgs::Path globalGpsPath;
    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    //  GpsTools gpsTools;
    //  Eigen::Vector3d optimized_lla;
    //  std::vector<Eigen::Vector3d> lla_vector;

    //  string savePCDDirectory;
    //  string scene_name;

    mapOptimization() {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        if (useGPS) {
            pubGPSOdometry = nh.advertise<sensor_msgs::NavSatFix>("lio_sam_6axis/mapping/odometry_gps", 1);
        }

        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>(
                "lio_sam_6axis/mapping/trajectory", 1);
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>(
                "lio_sam_6axis/mapping/map_global", 1);
        pubLaserOdometryGlobal =
                nh.advertise<nav_msgs::Odometry>("lio_sam_6axis/mapping/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry>(
                "lio_sam_6axis/mapping/odometry_incremental", 1);
        pubPath = nh.advertise<nav_msgs::Path>("/path", 1);

        subCloud = nh.subscribe<lio_sam_6axis::cloud_info>(
                "lio_sam_6axis/feature/cloud_info", 1,
                &mapOptimization::laserCloudInfoHandler, this,
                ros::TransportHints().tcpNoDelay());

        subGPS = nh.subscribe<nav_msgs::Odometry>(
                "gps_odom", 200, &mapOptimization::gpsHandler, this,
                ros::TransportHints().tcpNoDelay());
        // 订阅路侧lidar配准的结果
        subRoadSide = nh.subscribe<lio_sam_6axis::road_registration>(
                    "lio_sam_6axis/roadside/registration", 200, &mapOptimization::RoadSideHandler, this,
                    ros::TransportHints().tcpNoDelay());

        subLoop = nh.subscribe<std_msgs::Float64MultiArray>(
                "lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler,
                this, ros::TransportHints().tcpNoDelay());
        srvSaveMap = nh.advertiseService("lio_sam_6axis/save_map",
                                         &mapOptimization::saveMapService, this);

        srvTransformClient = nh.serviceClient<lio_sam_6axis::transformInterp>("transform_interp_service");

        // 启动异步服务检查线程 [!++]
        serviceCheckThread = std::thread([this]() {
            bool success = ros::service::waitForService("transform_interp_service", ros::Duration(3.0));
            serviceAvailable.store(success);  // [!++ 小驼峰变量]
            
            if (success) {
                ROS_INFO("Transform service connected");
            } else {
                ROS_INFO("Transform service unavailable after 3.0 seconds!");
            }
        });


        pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>(
                "lio_sam_6axis/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>(
                "lio_sam_6axis/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>(
                "/lio_sam_6axis/mapping/loop_closure_constraints", 1);
        pubGpsConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>(
                "/lio_sam_6axis/mapping/gps_constraints", 1);

        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>(
                "lio_sam_6axis/mapping/map_local", 1);
        pubRecentKeyFrame = nh.advertise<sensor_msgs::PointCloud2>(
                "lio_sam_6axis/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>(
                "lio_sam_6axis/mapping/cloud_registered_raw", 1);
        pubCloudRaw = nh.advertise<sensor_msgs::PointCloud2>("cloud_deskewed", 1);

        pubSLAMInfo = nh.advertise<lio_sam_6axis::cloud_info>(
                "lio_sam_6axis/mapping/slam_info", 1);

        pubHessianMatrix = nh.advertise<std_msgs::Float32MultiArray>("lio_sam_6axis/mapping/hessian_matrix", 50);
        pubGNSSOriginLLA = nh.advertise<geometry_msgs::Vector3>("lio_sam_6axis/mapping/originLLA", 20);

        downSizeFilterCorner.setLeafSize(
                mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize,
                                       mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize,
                                      mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(
                surroundingKeyframeDensity, surroundingKeyframeDensity,
                surroundingKeyframeDensity);  // for surrounding key poses of
        // scan-to-map optimization

        const float rawMapFilterSize = 0.5;  // giseop
        downSizeFilterRaw.setLeafSize(rawMapFilterSize, rawMapFilterSize,
                                      rawMapFilterSize);  // giseop

        // set log dir
        dataSaverPtr = std::make_unique<DataSaver>(saveDirectory, sequence);
        // use imu frame when saving map
        dataSaverPtr->setExtrinc(true, t_body_sensor, q_body_sensor);
        dataSaverPtr->setConfigDir(configDirectory);

        allocateMemory();
        std::cout << savePCDDirectory << std::endl;
        std::cout << sequence << std::endl;
    }

    void allocateMemory() {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyGPSPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses2D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(
                new pcl::PointCloud<PointType>());  // corner feature set from
        // odoOptimization
        laserCloudSurfLast.reset(
                new pcl::PointCloud<PointType>());  // surf feature set from
        // odoOptimization
        laserCloudCornerLastDS.reset(
                new pcl::PointCloud<PointType>());  // downsampled corner featuer set
        // from odoOptimization
        laserCloudSurfLastDS.reset(
                new pcl::PointCloud<PointType>());  // downsampled surf featuer set from
        // odoOptimization

        laserCloudRaw.reset(new pcl::PointCloud<PointType>());    // giseop
        laserCloudRawDS.reset(new pcl::PointCloud<PointType>());  // giseop

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(),
                  false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(),
                  false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i) {
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }

    void laserCloudInfoHandler(const lio_sam_6axis::cloud_infoConstPtr &msgIn) {
        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();

        // extract info and feature cloud
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);
        pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudRaw);  // 当前关键帧经过去畸变的点云

        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval) {
            timeLastProcessing = timeLaserInfoCur;
            PrintTransformTobeMapped("before updateInitialGuess Finished");
            updateInitialGuess();
            PrintTransformTobeMapped("after updateInitialGuess Finished");
            if (systemInitialized) {
                extractSurroundingKeyFrames();
                // 对当前帧进行下采样
                downsampleCurrentScan();
                // 当前帧和地图进行匹配
                scan2MapOptimization();
                // 根据配准结果确定当前是否为关键帧
                saveKeyFramesAndFactor();
                // 调整全局的轨迹
                correctPoses();
                // 将lidar里程计的信息发送出去
                publishOdometry();
                // 发布可视化点云信息
                publishFrames();
            }
        }
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr &gpsMsg) {
        if (useGPS) {
            mtxGpsInfo.lock();
            gpsQueue.push_back(*gpsMsg);
            mtxGpsInfo.unlock();
        }
    }


    void RoadSideHandler(const lio_sam_6axis::road_registrationConstPtr &roadSideMsg) {
        if (useRoadSide) {
            std::lock_guard<std::mutex> lock(mtxRoadSideInfo);
            roadRegistrationQueue.push_back(*roadSideMsg);
            if (debugRoadSide) {
                ROS_INFO("[roadRegistrationQueue] size : %d", roadRegistrationQueue.size());
            }
            
        }
    }

    void pointAssociateToMap(PointType const *const pi, PointType *const po) {
        po->x = transPointAssociateToMap(0, 0) * pi->x +
                transPointAssociateToMap(0, 1) * pi->y +
                transPointAssociateToMap(0, 2) * pi->z +
                transPointAssociateToMap(0, 3);
        po->y = transPointAssociateToMap(1, 0) * pi->x +
                transPointAssociateToMap(1, 1) * pi->y +
                transPointAssociateToMap(1, 2) * pi->z +
                transPointAssociateToMap(1, 3);
        po->z = transPointAssociateToMap(2, 0) * pi->x +
                transPointAssociateToMap(2, 1) * pi->y +
                transPointAssociateToMap(2, 2) * pi->z +
                transPointAssociateToMap(2, 3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(
            pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn) {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(
                transformIn->x, transformIn->y, transformIn->z, transformIn->roll,
                transformIn->pitch, transformIn->yaw);

#pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i) {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0, 0) * pointFrom.x +
                                    transCur(0, 1) * pointFrom.y +
                                    transCur(0, 2) * pointFrom.z + transCur(0, 3);
            cloudOut->points[i].y = transCur(1, 0) * pointFrom.x +
                                    transCur(1, 1) * pointFrom.y +
                                    transCur(1, 2) * pointFrom.z + transCur(1, 3);
            cloudOut->points[i].z = transCur(2, 0) * pointFrom.x +
                                    transCur(2, 1) * pointFrom.y +
                                    transCur(2, 2) * pointFrom.z + transCur(2, 3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
        return gtsam::Pose3(
                gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch),
                                    double(thisPoint.yaw)),
                gtsam::Point3(double(thisPoint.x), double(thisPoint.y),
                              double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[]) {
        return gtsam::Pose3(
                gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint) {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z,
                                      thisPoint.roll, thisPoint.pitch,
                                      thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[]) {
        return pcl::getTransformation(transformIn[3], transformIn[4],
                                      transformIn[5], transformIn[0],
                                      transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[]) {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw = transformIn[2];
        return thisPose6D;
    }

    bool syncGPS(std::deque<nav_msgs::Odometry> &gpsBuf,
                 nav_msgs::Odometry &aligedGps, double timestamp,
                 double eps_cam) {
        bool hasGPS = false;
        while (!gpsQueue.empty()) {
            mtxGpsInfo.lock();
            if (gpsQueue.front().header.stamp.toSec() < timestamp - eps_cam) {
                // message too old
                gpsQueue.pop_front();
                mtxGpsInfo.unlock();
            } else if (gpsQueue.front().header.stamp.toSec() > timestamp + eps_cam) {
                // message too new
                mtxGpsInfo.unlock();
                break;
            } else {
                hasGPS = true;
                aligedGps = gpsQueue.front();
                gpsQueue.pop_front();
//                if (debugGps)
//                    ROS_INFO("GPS time offset %f ",
//                             aligedGps.header.stamp.toSec() - timestamp);
                mtxGpsInfo.unlock();
            }
        }

        if (hasGPS)
            return true;
        else
            return false;
    }

    bool saveMapService(std_srvs::Empty::Request &req,
                        std_srvs::Empty::Response &res) {
        if (cloudKeyPoses6D->size() < 1) {
            ROS_INFO("NO ENCOUGH POSE!");
            return false;
        }

        Eigen::Vector3d optimized_lla;
        if (useGPS) {
            Eigen::Vector3d first_point(cloudKeyPoses6D->at(0).x,
                                        cloudKeyPoses6D->at(0).y,
                                        cloudKeyPoses6D->at(0).z);
            // we save optimized origin gps point
            geo_converter.Reverse(first_point[0], first_point[1], first_point[2], optimized_lla[0], optimized_lla[1],
                                  optimized_lla[2]);
            std::cout << std::setprecision(9)
                      << "origin LLA: " << originLLA.transpose() << std::endl;
            std::cout << std::setprecision(9)
                      << "optimized LLA: " << optimized_lla.transpose() << std::endl;
            dataSaverPtr->saveOriginGPS(optimized_lla);
        }
        geo_converter.Reset(optimized_lla[0], optimized_lla[1], optimized_lla[2]);

        vector<pcl::PointCloud<PointType>::Ptr> keyframePc;
        std::vector<Eigen::Vector3d> lla_vec;
        std::vector<geometry_msgs::TransformStamped> transform_vec;
        // pcl::PointCloud<PointType>::Ptr temp_ptr(new pcl::PointCloud<PointType>);
        std::vector<nav_msgs::Odometry> keyframePosesOdom;

        for (int i = 0; i < cloudKeyPoses6D->size(); ++i) {
            mtx.lock();
            PointTypePose p = cloudKeyPoses6D->at(i);
            mtx.unlock();

            // keyframeTimes.push_back(p.time);
            //            Eigen::Translation3d tf_trans(p.x, p.y, p.z);
            //            Eigen::AngleAxisd rot_x(p.roll, Eigen::Vector3d::UnitX());
            //            Eigen::AngleAxisd rot_y(p.pitch, Eigen::Vector3d::UnitY());
            //            Eigen::AngleAxisd rot_z(p.yaw, Eigen::Vector3d::UnitZ());
            //            Eigen::Matrix4d trans = (tf_trans * rot_z * rot_y *
            //            rot_x).matrix(); keyframePosestrans.push_back(trans);

            nav_msgs::Odometry laserOdometryROS;
            laserOdometryROS.header.stamp = ros::Time().fromSec(p.time);
            laserOdometryROS.header.frame_id = odometryFrame;
            laserOdometryROS.child_frame_id = lidarFrame;
            laserOdometryROS.pose.pose.position.x = p.x;
            laserOdometryROS.pose.pose.position.y = p.y;
            laserOdometryROS.pose.pose.position.z = p.z;
            laserOdometryROS.pose.pose.orientation =
                    tf::createQuaternionMsgFromRollPitchYaw(p.roll, p.pitch, p.yaw);
            keyframePosesOdom.push_back(laserOdometryROS);

            //            temp_ptr->clear();
            //            *temp_ptr += *surfCloudKeyFrames.at(i);
            //            *temp_ptr += *cornerCloudKeyFrames.at(i);
            //            keyframePc.push_back(temp_ptr);

            geometry_msgs::TransformStamped transform_msg;
            transform_msg.header.stamp = ros::Time().fromSec(p.time);
            transform_msg.header.frame_id = odometryFrame;
            transform_msg.child_frame_id = lidarFrame;
            transform_msg.transform.translation.x = p.x;
            transform_msg.transform.translation.y = p.y;
            transform_msg.transform.translation.z = p.z;
            transform_msg.transform.rotation =
                    tf::createQuaternionMsgFromRollPitchYaw(p.roll, p.pitch, p.yaw);
            transform_vec.push_back(transform_msg);

            if (useGPS) {
                // we save optimized origin gps point, maybe the altitude value need to
                // be fixes
                Eigen::Vector3d first_point(p.x, p.y, p.z);
                Eigen::Vector3d lla_point;
                geo_converter.Reverse(first_point[0], first_point[1], first_point[2], lla_point[0], lla_point[1],
                                      lla_point[2]);
                lla_vec.push_back(lla_point);
            }
        }

        dataSaverPtr->saveTimes(keyframeTimes);
        dataSaverPtr->saveOptimizedVerticesTUM(isamCurrentEstimate);
        //dataSaverPtr->saveOptimizedVerticesKITTI(isamCurrentEstimate);
        dataSaverPtr->saveOdometryVerticesTUM(keyframeRawOdom);
        // dataSaverPtr->saveOdometrycloudKeyPoses6DTUM(cloudKeyPoses6D);

        // dataSaverPtr->saveResultBag(keyframePosesOdom, keyframeCloudDeskewed, transform_vec);
        if (useGPS) dataSaverPtr->saveKMLTrajectory(lla_vec);

        /** always remember do not call this service if your databag do not play over!!!!!!!!*/
        mtxGraph.lock();
        dataSaverPtr->saveGraphGtsam(gtSAMgraph, isam, isamCurrentEstimate);
        mtxGraph.unlock();

        std::cout << "Times, isam, raw_odom, pose_odom, pose3D, pose6D: "
                  << keyframeTimes.size() << " " << isamCurrentEstimate.size()
                  << ", " << keyframeRawOdom.size() << " "
                  << keyframePosesOdom.size() << " " << cloudKeyPoses3D->size()
                  << " " << cloudKeyPoses6D->size() << std::endl;
        std::cout << "key_cloud, surf, corner, raw_frame size: "
                  << keyframePc.size() << " " << surfCloudKeyFrames.size() << " "
                  << cornerCloudKeyFrames.size() << " "
                  << laserCloudRawKeyFrames.size() << std::endl;

        pcl::PointCloud<PointType>::Ptr globalCornerCloud(
                new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(
                new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(
                new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(
                new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalRawCloud(
                new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalRawCloudDS(
                new pcl::PointCloud<PointType>());

        pcl::PointCloud<PointType>::Ptr globalMapCloud(
                new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int) cloudKeyPoses3D->size(); i++) {
            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],
                                                       &cloudKeyPoses6D->points[i]);
            *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i],
                                                     &cloudKeyPoses6D->points[i]);
            /** if you want to save the origin deskewed point cloud, but not only feature map*/
           *globalRawCloud += *transformPointCloud(laserCloudRawKeyFrames[i],
                                                   &cloudKeyPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of "
                 << cloudKeyPoses6D->size() << " ..." << std::endl;
        }

        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.setLeafSize(globalMapLeafSize, globalMapLeafSize, globalMapLeafSize);
        downSizeFilterCorner.filter(*globalCornerCloudDS);

        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(globalMapLeafSize, globalMapLeafSize, globalMapLeafSize);
        downSizeFilterSurf.filter(*globalSurfCloudDS);

        // save global point cloud map
        *globalMapCloud += *globalCornerCloudDS;
        *globalMapCloud += *globalSurfCloudDS;

        // 保存原始点云，但是点云转换到了全局坐标系中，不保存到最终的地图中，也就是说最终的地图只保存面和线的特征点
        // *globalMapCloud += *globalRawCloud;

       downSizeFilterSurf.setInputCloud(globalRawCloud);
       downSizeFilterSurf.setLeafSize(globalMapLeafSize, globalMapLeafSize,
                                      globalMapLeafSize);
       downSizeFilterSurf.filter(*globalRawCloudDS);

        /** if you need to downsample the final map, 0.5m is ok*/
        downSizeFilterSurf.setInputCloud(globalMapCloud);
        downSizeFilterSurf.setLeafSize(globalMapLeafSize, globalMapLeafSize, globalMapLeafSize);
        downSizeFilterSurf.filter(*globalMapCloud);


        std::cout << "global map size: " << globalMapCloud->size() << " " << globalRawCloudDS->size() << std::endl;
        dataSaverPtr->savePointCloudMap(*globalMapCloud);
//         dataSaverPtr->savePointCloudMap(keyframePosesOdom,
//         laserCloudRawKeyFrames);

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed: " << endl;

        return true;
    }

    void visualizeGlobalMapThread() {
        ros::Rate rate(0.2);
        while (ros::ok()) {
            rate.sleep();
            publishGlobalMap();
        }
    }

    void publishGlobalMap() {
        if (pubLaserCloudSurround.getNumSubscribers() == 0) return;

        if (cloudKeyPoses3D->points.empty() == true) return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(
                new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(
                new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(
                new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(
                new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(
                new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(
                cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius,
                pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int) pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(
                    cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType>
                downSizeFilterGlobalMapKeyPoses;  // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(
                globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity,
                globalMapVisualizationPoseDensity);  // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for (auto &pt : globalMapKeyPosesDS->points) {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap,
                                            pointSearchSqDisGlobalMap);
            pt.intensity =
                    cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int) globalMapKeyPosesDS->size(); ++i) {
            if (pointDistance(globalMapKeyPosesDS->points[i],
                              cloudKeyPoses3D->back()) >
                globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int) globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames +=
                    *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],
                                         &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(
                    surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType>
                downSizeFilterGlobalMapKeyFrames;  // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(
                globalMapVisualizationLeafSize, globalMapVisualizationLeafSize,
                globalMapVisualizationLeafSize);  // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS,
                     timeLaserInfoStamp, odometryFrame);
    }

    void loopClosureThread() {
        // 如果不需要进行回环检测，则直接返回
        if (loopClosureEnableFlag == false) return;

        ros::Rate rate(loopClosureFrequency);
        while (ros::ok()) {
            ros::spinOnce();
            // 执行回环检测
            performLoopClosure();
            visualizeLoopClosure();

            if (useGPS) visualGPSConstraint();

            rate.sleep();
        }
    }

    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr &loopMsg) {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2) return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5) loopInfoVec.pop_front();
    }

    void performLoopClosure() {
        if (cloudKeyPoses3D->points.empty() == true) return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses2D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            // 根据里程计的距离来检测回环
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false) return;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(
                new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(
                new pcl::PointCloud<PointType>());
        {
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre,
                                  historyKeyframeSearchNum);
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp,
                             odometryFrame);
        }

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(
                new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (icp.hasConverged() == false ||
            icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // publish corrected cloud
        if (pubIcpKeyFrames.getNumSubscribers() != 0) {
            pcl::PointCloud<PointType>::Ptr closed_cloud(
                    new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud,
                                     icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp,
                         odometryFrame);
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong =
                pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose  修正后的当前帧的位姿
        Eigen::Affine3f tCorrect =
                correctionLidarFrame *
                tWrong;  // pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
        // 当前修正后的位姿，给转换成了gtsam的格式
        gtsam::Pose3 poseFrom =
                Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        // 回环帧的位姿，也就是之前的位姿
        gtsam::Pose3 poseTo =
                pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        // icp的得分作为它的噪声
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore,
                noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise =
                noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant 保存已经配对的约束对
        loopIndexContainer[loopKeyCur] = loopKeyPre;
        lastLoopIndex = loopKeyCur;
    }

    bool detectLoopClosureDistance(int *latestID, int *closestID) {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end()) return false;

        // tricks
        // Two consecutive loop edges represent the closed loop of the same scene.
        // Adding all of them to the pose graph has little meaning and may reduce
        // the accuracy.
        if (abs(lastLoopIndex - loopKeyCur) < 5 && lastLoopIndex != -1)
            return false;

        // tricks
        // sometimes we need to find the corressponding loop pairs
        // but we do not need to care about the z values of these poses.
        // Pls note that this is not work for stair case
        for (int i = 0; i < copy_cloudKeyPoses2D->size(); ++i) {
            copy_cloudKeyPoses2D->at(i).z = 0;
        }

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses2D);
        kdtreeHistoryKeyPoses->radiusSearch(
                copy_cloudKeyPoses2D->back(), historyKeyframeSearchRadius,
                pointSearchIndLoop, pointSearchSqDisLoop, 0);

        for (int i = 0; i < (int) pointSearchIndLoop.size(); ++i) {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) >
                historyKeyframeSearchTimeDiff) {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre) return false;

        // we also need to care about the accumulated distance between keyframe;
        // LOOPs that are too close together have no meaning and may reduce
        // accuracy. For example, the lidar starts to move after being stationary
        // for 30s in a certain place. At this time, the IMU should be trusted more
        // than the lidar.
        if (keyframeDistances.size() >= loopKeyCur) {
            double distance = 0.0;
            for (int j = loopKeyPre; j < loopKeyCur; ++j) {
                distance += keyframeDistances.at(j);
            }
            if (distance < 12) {
                std::cout << "CLOSE FRAME MUST FILTER OUT " << distance << std::endl;
                return false;
            }
        }

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    bool detectLoopClosureExternal(int *latestID, int *closestID) {
        // this function is not used yet, please ignore it
        // 作者说这个函数还没有使用，所以请忽略
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty()) return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2) return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i) {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i) {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre) return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end()) return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        ROS_INFO("VISUAL LOOP INDEX: %d, %d", loopKeyCur, loopKeyPre);

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes,
                               const int &key, const int &searchNum) {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i) {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize) continue;
            *nearKeyframes +=
                    *transformPointCloud(cornerCloudKeyFrames[keyNear],
                                         &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(
                    surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty()) return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(
                new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void visualizeLoopClosure() {
        if (loopIndexContainer.empty()) return;

        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.15;
        markerNode.scale.y = 0.15;
        markerNode.scale.z = 0.15;
        markerNode.color.r = 0;
        markerNode.color.g = 0.8;
        markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9;
        markerEdge.color.g = 0.9;
        markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end();
             ++it) {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }

    void visualGPSConstraint() {
        if (gpsIndexContainer.empty()) return;

        visualization_msgs::MarkerArray markerArray;
        // gps nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "gps_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3;
        markerNode.scale.y = 0.3;
        markerNode.scale.z = 0.3;
        markerNode.color.r = 0.8;
        markerNode.color.g = 0;
        markerNode.color.b = 1;
        markerNode.color.a = 1;

        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "gps_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.2;
        markerEdge.color.r = 0.9;
        markerEdge.color.g = 0;
        markerEdge.color.b = 0.1;
        markerEdge.color.a = 1;

        for (auto it = gpsIndexContainer.begin(); it != gpsIndexContainer.end();
             ++it) {
            int key_cur = it->first;
            int key_pre = it->second;

            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);

            p.x = cloudKeyGPSPoses3D->points[key_pre].x;
            p.y = cloudKeyGPSPoses3D->points[key_pre].y;
            p.z = cloudKeyGPSPoses3D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubGpsConstraintEdge.publish(markerArray);
    }

    // 调用python脚本获取指定时间戳的位姿变换矩阵
    /**
     * @brief 访问服务并获取指定时间戳的4x4变换矩阵
     * @param timestamp 输入时间戳
     * @return Eigen::Matrix4d 4x4齐次变换矩阵（直接可用作位姿变换）
     */
    Eigen::Matrix4d GetTransformFromPython(double timestamp) {

        // 1. 服务可用性检查（建议放在构造函数）
        if (!srvTransformClient.exists()) {
            ROS_WARN_THROTTLE(1.0, "Waiting for transform_interp_service...");
            if (!srvTransformClient.waitForExistence(ros::Duration(2.0))) {
                ROS_ERROR("Service [transform_interp_service] not available");
                return Eigen::Matrix4d::Identity();
            }
        }

        lio_sam_6axis::transformInterp srv;
        srv.request.timestamp = timestamp;
    
        const int max_retries = 2;
        for (int retry = 0; retry < max_retries; ++retry) {  // [!++ 重试max_retries次]
            try {
                // 带超时的服务调用
                if (!srvTransformClient.call(srv)) {
                    ROS_ERROR_THROTTLE(1.0, "Service call failed (attempt %d/%d)", retry+1, max_retries);
                    continue;
                }
    
                // 响应数据校验
                if (srv.response.matrix.size() != 16) {
                    ROS_ERROR_STREAM("Invalid matrix size: " << srv.response.matrix.size());
                    return Eigen::Matrix4d::Identity();
                }
    
                // 安全的内存映射（禁用对齐）
                const Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>, Eigen::Unaligned> mat(
                    srv.response.matrix.data(),
                    4, 4
                );
                return mat;
    
            } catch (const ros::Exception& e) {
                ROS_ERROR_STREAM("ROS exception: " << e.what());
            } catch (const std::exception& e) {
                ROS_ERROR_STREAM("System error: " << e.what());
            } catch (...) {
                ROS_WARN("Unknown exception during service call");
            }
    
            if (retry < max_retries - 1) {
                ros::Duration(0.5).sleep();
            }
        }
    
        ROS_ERROR("All service attempts failed for timestamp: %.3f", timestamp);
        return Eigen::Matrix4d::Identity();
    }
    void PublishOriginGNSS(Eigen::Vector3d origin) {
        geometry_msgs::Vector3 msg;
        msg.x = origin[0];
        msg.y = origin[1];
        msg.z = origin[2];
        pubGNSSOriginLLA.publish(msg);
        return;
    }

    void ComputeTransform(double timestamp, float outputTransform[6]) {
        auto start = std::chrono::steady_clock::now(); // 开始计时
        
        // 通过 ROS 服务（或其他接口）获取对应时间戳的4x4变换矩阵
        Eigen::Matrix4d InitialTransform = GetTransformFromPython(timestamp);

        tf::Matrix3x3 tfRot;
        tfRot.setValue(
            InitialTransform(0, 0), InitialTransform(0, 1), InitialTransform(0, 2),
            InitialTransform(1, 0), InitialTransform(1, 1), InitialTransform(1, 2),
            InitialTransform(2, 0), InitialTransform(2, 1), InitialTransform(2, 2)
        );
        double roll, pitch, yaw;
        tfRot.getRPY(roll, pitch, yaw);
        double x = InitialTransform(0, 3);
        double y = InitialTransform(1, 3);
        double z = InitialTransform(2, 3);
        
        // 将结果存入输出数组：前3个为旋转，后3个为平移
        outputTransform[0] = static_cast<float>(roll);
        outputTransform[1] = static_cast<float>(pitch);
        outputTransform[2] = static_cast<float>(yaw);
        outputTransform[3] = static_cast<float>(x);
        outputTransform[4] = static_cast<float>(y);
        outputTransform[5] = static_cast<float>(z);
        
        auto end = std::chrono::steady_clock::now(); // 结束计时
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        if (debugInitialVehiclePose) {
            ROS_INFO_STREAM("ComputeTransform executed in " << duration << " ms");
        }
    }

    // 将上一帧lidar最优估计的位姿(transformTobeMapped)转换到了当前帧lidar位姿，这个位姿后面要拿来作为当前帧的初始位姿
    void updateInitialGuess() {
        // save current transformation before any processing  上一帧isam优化后的最佳位姿
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);
        static Eigen::Affine3f lastImuTransformation;

        // initialization the first frame
        if (cloudKeyPoses3D->points.empty()) {
            systemInitialized = false;
            if (useGPS) {
                ROS_INFO("GPS use to init pose");
                /** when you align gnss and lidar timestamp, make sure (1.0/gpsFrequence) is small encougn
                 *  no need to care about the real gnss frquency. time alignment fail will cause
                 *  "[ERROR] [1689196991.604771416]: sysyem need to be initialized"
                 * */
                nav_msgs::Odometry alignedGPS;
                if (syncGPS(gpsQueue, alignedGPS, timeLaserInfoCur, 1.0 / gpsFrequence)) {
                    /** we store the origin wgs84 coordinate points in covariance[1]-[3] */
                    originLLA.setIdentity();
                    originLLA = Eigen::Vector3d(alignedGPS.pose.covariance[1],
                                                alignedGPS.pose.covariance[2],
                                                alignedGPS.pose.covariance[3]);
                    PublishOriginGNSS(originLLA);
                    ROS_INFO_STREAM("[updateInitialGuess] originLLA: " << std::fixed << std::setprecision(9) << originLLA.transpose());                    /** set your map origin points */
                    geo_converter.Reset(originLLA[0], originLLA[1], originLLA[2]);
                    // WGS84->ENU, must be (0,0,0)
                    Eigen::Vector3d enu;
                    geo_converter.Forward(originLLA[0], originLLA[1], originLLA[2], enu[0], enu[1], enu[2]);

                    if (debugGps) {
                        double roll, pitch, yaw;
                        tf::Matrix3x3(tf::Quaternion(alignedGPS.pose.pose.orientation.x,
                                                     alignedGPS.pose.pose.orientation.y,
                                                     alignedGPS.pose.pose.orientation.z,
                                                     alignedGPS.pose.pose.orientation.w))
                                .getRPY(roll, pitch, yaw);
                        std::cout << "initial gps yaw: " << yaw << std::endl;
                        std::cout << "GPS Position: " << enu.transpose() << std::endl;
                        std::cout << "GPS LLA: " << originLLA.transpose() << std::endl;
                    }

                    /** add the first factor, we need this origin GPS point for prior map based localization,
                     * but we need to optimize its value by pose graph if the origin gps RTK status is not fixed.*/
                    PointType gnssPoint;
                    gnssPoint.x = enu[0],
                    gnssPoint.y = enu[1],
                    gnssPoint.z = enu[2];
                    float noise_x = alignedGPS.pose.covariance[0];
                    float noise_y = alignedGPS.pose.covariance[7];
                    float noise_z = alignedGPS.pose.covariance[14];

                    /** if we get reliable origin point, we adjust the weight of this gps factor to fix the map origin */
                    //if (!updateOrigin) {
                    noise_x *= 1e-4;
                    noise_y *= 1e-4;
                    noise_z *= 1e-4;
                    // }
                    gtsam::Vector Vector3(3);
                    Vector3 << noise_x, noise_y, noise_z;
                    noiseModel::Diagonal::shared_ptr gps_noise =
                            noiseModel::Diagonal::Variances(Vector3);
                    gtsam::GPSFactor gps_factor(0, gtsam::Point3(gnssPoint.x, gnssPoint.y, gnssPoint.z),
                                                gps_noise);
                    keyframeGPSfactor.push_back(gps_factor);
                    cloudKeyGPSPoses3D->points.push_back(gnssPoint);

                    transformTobeMapped[0] = cloudInfo.imuRollInit;
                    transformTobeMapped[1] = cloudInfo.imuPitchInit;
                    transformTobeMapped[2] = cloudInfo.imuYawInit;
                    if (!useImuHeadingInitialization) transformTobeMapped[2] = 0;
                    lastImuTransformation = pcl::getTransformation(
                            0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit,
                            cloudInfo.imuYawInit);
                    systemInitialized = true;
                    ROS_WARN("GPS init success");
                }
            } else if(uesInitialVehiclePose) {
                // 通过ros服务获取第一帧的位姿
                ComputeTransform(timeLaserInfoCur, transformTobeMapped);
                lastImuTransformation = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                    transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
                systemInitialized = true;
                return;
            }
            else {
                transformTobeMapped[0] = cloudInfo.imuRollInit;
                transformTobeMapped[1] = cloudInfo.imuPitchInit;
                transformTobeMapped[2] = cloudInfo.imuYawInit;
                // 这里认为默认不使用yaw角，将yaw置为0
                if (!useImuHeadingInitialization) transformTobeMapped[2] = 0;

                lastImuTransformation = pcl::getTransformation(
                        0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit,
                        cloudInfo.imuYawInit);

                systemInitialized = true;
                return;
            }

        }

        if (!systemInitialized) {
            ROS_ERROR("sysyem need to be initialized");
            return;
        }

        // if not the first frame
        // use imu pre-integration estimation for pose guess
        PrintTransformTobeMapped("before cloudInfo.odomAvailable Finished");
        static bool lastImuPreTransAvailable = false;
        static int  odomNum = 0;
        static Eigen::Affine3f lastImuPreTransformation;
        // 预积分节点提供的里程计信息
        if (cloudInfo.odomAvailable == true) {
            if(debugInitialVehiclePose) {
                odomNum ++;
                ROS_INFO_STREAM("odomNum call count: " << odomNum);
                ROS_INFO_STREAM("Initial Guess: roll=" << cloudInfo.initialGuessRoll << " pitch=" << cloudInfo.initialGuessPitch
                    << " yaw=" << cloudInfo.initialGuessYaw << " x=" << cloudInfo.initialGuessX
                    << " y=" << cloudInfo.initialGuessY << " z=" << cloudInfo.initialGuessZ);
            }
            Eigen::Affine3f transBack = pcl::getTransformation(
                    cloudInfo.initialGuessX, cloudInfo.initialGuessY,
                    cloudInfo.initialGuessZ, cloudInfo.initialGuessRoll,
                    cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            if (lastImuPreTransAvailable == false) {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            } else {
                // 计算上一个预积分的位姿和当前预积分的位姿之间的delta pose，也就是两个关键帧之间的位姿变换transIncre
                Eigen::Affine3f transIncre =
                        lastImuPreTransformation.inverse() * transBack;
                // 上一个lidar关键帧的位姿估计，这个位姿估计很准确
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                // 得到当前关键帧的先验估计位姿
                Eigen::Affine3f transFinal = transTobe * transIncre;
                // 将先验位姿估计转换成平移和欧拉角
                pcl::getTranslationAndEulerAngles(
                        transFinal, transformTobeMapped[3], transformTobeMapped[4],
                        transformTobeMapped[5], transformTobeMapped[0],
                        transformTobeMapped[1], transformTobeMapped[2]);
                // 将当前帧的预积分位姿保存下来，方便下次使用
                lastImuPreTransformation = transBack;
                // 虽然有lidar里程计，但是还是将磁力计的姿态角给保存下来了
                lastImuTransformation = pcl::getTransformation(
                        0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit,
                        cloudInfo.imuYawInit);  // save imu before return;
                PrintTransformTobeMapped("lastImuPreTransAvailable is true, cloudInfo.odomAvailable Finished");
                return;
            }
        }
        PrintTransformTobeMapped("lastImuPreTransAvailable is false, cloudInfo.odomAvailable Finished");

        // use imu incremental estimation for pose guess (only rotation)
        // 如果没有imu的预积分里程计的信息，就使用imu的旋转信息来更新，作者认为单独使用imu无法得到靠谱的平移信息，因此平移置为0
        static int  getServiceNum = 0;
        if (cloudInfo.imuAvailable == true) {
            float initialGuess[6] = {0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit}; 
            if(uesInitialVehiclePose) {
                ComputeTransform(timeLaserInfoCur, initialGuess);
                getServiceNum ++;
                ROS_INFO_STREAM("Service call count: " << getServiceNum);
            }
            Eigen::Affine3f transBack =
                    pcl::getTransformation(initialGuess[3], initialGuess[4], initialGuess[5], initialGuess[0],
                        initialGuess[1], initialGuess[2]);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(
                    transFinal, transformTobeMapped[3], transformTobeMapped[4],
                    transformTobeMapped[5], transformTobeMapped[0],
                    transformTobeMapped[1], transformTobeMapped[2]);

            // update last imu transformation
            lastImuTransformation = pcl::getTransformation(initialGuess[3], initialGuess[4], initialGuess[5], initialGuess[0],
                initialGuess[1], initialGuess[2]);  // save imu before return;
            PrintTransformTobeMapped("after cloudInfo.imuAvailable Finished");
            return;
        }

    }

    void extractForLoopClosure() {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(
                new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses - 1; i >= 0; --i) {
            if (cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    void extractNearby() {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(
                new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(
                new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(
                cloudKeyPoses3D);  // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(
                cloudKeyPoses3D->back(), (double) surroundingKeyframeSearchRadius,
                pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int) pointSearchInd.size(); ++i) {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        for (auto &pt : surroundingKeyPosesDS->points) {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd,
                                                      pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one
        // position
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses - 1; i >= 0; --i) {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }
        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract) {
        // fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        for (int i = 0; i < (int) cloudToExtract->size(); ++i) {
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) >
                surroundingKeyframeSearchRadius)
                continue;

            int thisKeyInd = (int) cloudToExtract->points[i].intensity;
            if (laserCloudMapContainer.find(thisKeyInd) !=
                laserCloudMapContainer.end()) {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                // 如果这个点云没有存储，那就需要将该帧对应的位姿，把该帧点云从当前帧的位姿转换到世界坐标系下
                pcl::PointCloud<PointType> laserCloudCornerTemp =
                        *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],
                                             &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp =
                        *transformPointCloud(surfCloudKeyFrames[thisKeyInd],
                                             &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap += laserCloudSurfTemp;
                // 把转换后的面点和角点存到这个容器中，存储的是面点和角点的指针，方便后续直接加入点云地图使用。
                laserCloudMapContainer[thisKeyInd] =
                        make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
        }

        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        if (laserCloudMapContainer.size() > 1000) laserCloudMapContainer.clear();
    }

    void extractSurroundingKeyFrames() {
        if (cloudKeyPoses3D->points.empty() == true) return;

        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();
        // } else {
        //     extractNearby();
        // }

        extractNearby();
    }

    void downsampleCurrentScan() {
        laserCloudRawDS->clear();
        downSizeFilterRaw.setInputCloud(laserCloudRaw);
        downSizeFilterRaw.filter(*laserCloudRawDS);

        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    void updatePointAssociateToMap() {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void cornerOptimization() {
        updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++) {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                                pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5;
                cy /= 5;
                cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax =
                            laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay =
                            laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az =
                            laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax;
                    a12 += ax * ay;
                    a13 += ax * az;
                    a22 += ay * ay;
                    a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5;
                a12 /= 5;
                a13 /= 5;
                a22 /= 5;
                a23 /= 5;
                a33 /= 5;

                matA1.at<float>(0, 0) = a11;
                matA1.at<float>(0, 1) = a12;
                matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12;
                matA1.at<float>(1, 1) = a22;
                matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13;
                matA1.at<float>(2, 1) = a23;
                matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 =
                            sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *
                                 ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                                 ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
                                 ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                                 ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
                                 ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

                    float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
                                     (z1 - z2) * (z1 - z2));

                    float la =
                            ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                             (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) /
                            a012 / l12;

                    float lb =
                            -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
                              (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
                            a012 / l12;

                    float lc =
                            -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                              (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
                            a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    void surfOptimization() {
        updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++) {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                              pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z +
                             pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) /
                                  sqrt(sqrt(pointOri.x * pointOri.x +
                                            pointOri.y * pointOri.y +
                                            pointOri.z * pointOri.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    void combineOptimizationCoeffs() {
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i) {
            if (laserCloudOriCornerFlag[i] == true) {
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i) {
            if (laserCloudOriSurfFlag[i] == true) {
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(),
                  false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(),
                  false);
    }

    // 发布海塞矩阵到python节点，方便接收使用
    void publishHessianMatrix(cv::Mat mat) {
        if (mat.type() != CV_32F || mat.rows != 6 || mat.cols != 6) {
            ROS_ERROR("Matrix must be 6x6 with type CV_32F.");
            return;
        }

        std_msgs::Float32MultiArray matrix_msg;
        matrix_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        matrix_msg.layout.dim[0].label = "rows";
        matrix_msg.layout.dim[0].size = mat.rows;
        matrix_msg.layout.dim[0].stride = mat.rows * mat.cols;

        matrix_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        matrix_msg.layout.dim[1].label = "cols";
        matrix_msg.layout.dim[1].size = mat.cols;
        matrix_msg.layout.dim[1].stride = mat.cols;

        matrix_msg.data.resize(mat.rows * mat.cols);
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                matrix_msg.data[i * mat.cols + j] = mat.at<float>(i, j);
            }
        }

        // 时间戳单位为秒，乘以1e4，因为data_offset类型为uint32_t，即为无符号整型
        matrix_msg.layout.data_offset = timeLaserInfoStamp.toSec() * 1e4;

        pubHessianMatrix.publish(matrix_msg);
    }


    bool LMOptimization(int iterCount) {
        // 原始的loam代码是将lidar坐标系转到相机坐标系，这里把原先loam中的代码拷贝了过来，但是为了坐标系的统一，就先转到相机系优化，然后结果转回lidar系
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera 将lidar系转到相机系
        float srx = sin(transformTobeMapped[1]); // transformTobeMapped[1]为lidar坐标系下的y pitch角
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]); // transformTobeMapped[2]为ldiar坐标系下的z  yaw角
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]); // transformTobeMapped[0]为lidar坐标系下的x roll角
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));  // A,也就是雅可比矩阵
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0)); // A的转置
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0)); // 海塞矩阵，就是A的转置乘以A
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            // 相机系下的旋转顺序是Y - X - Z对应lidar系下Z -Y -X
            // camera中的roll横滚角(对应camera的arx，因为roll的定义是横滚角，定义就是绕x轴旋转),对应lidar中的pitch
            float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y -
                         srx * sry * pointOri.z) *
                        coeff.x +
                        (-srx * srz * pointOri.x - crz * srx * pointOri.y -
                         crx * pointOri.z) *
                        coeff.y +
                        (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y -
                         cry * srx * pointOri.z) *
                        coeff.z; // 关于roll的导数
            // camera中的pitch俯仰角(对应camera的ary，pitch的定义就是绕y轴旋转),对应lidar中的yaw
            float ary = ((cry * srx * srz - crz * sry) * pointOri.x +
                         (sry * srz + cry * crz * srx) * pointOri.y +
                         crx * cry * pointOri.z) *
                        coeff.x +
                        ((-cry * crz - srx * sry * srz) * pointOri.x +
                         (cry * srz - crz * srx * sry) * pointOri.y -
                         crx * sry * pointOri.z) *
                        coeff.z;
            // camera中的yaw航向角(对应camera中的arz，yaw的定义就是绕z轴旋转),对应lidar中的roll
            float arz = ((crz * srx * sry - cry * srz) * pointOri.x +
                         (-cry * crz - srx * sry * srz) * pointOri.y) *
                        coeff.x +
                        (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y +
                        ((sry * srz + cry * crz * srx) * pointOri.x +
                         (crz * sry - cry * srx * srz) * pointOri.y) *
                        coeff.z;
            // camera -> lidar
            matA.at<float>(i, 0) = arz; // matA(i, 0)为lidar坐标系下的雅可比矩阵中的x，即为lidar系下的roll
            matA.at<float>(i, 1) = arx; // matA(i, 1)为lidar坐标系下的雅可比矩阵中的y,即为lidarx系下的pitch
            matA.at<float>(i, 2) = ary; // matA(i, 2)为lidar坐标系下的雅可比矩阵中的z,即为lidar系下的yaw
            matA.at<float>(i, 3) = coeff.z; // 雅可比矩阵中的x
            matA.at<float>(i, 4) = coeff.x; // 雅可比矩阵中的y
            matA.at<float>(i, 5) = coeff.y; // 雅可比矩阵中的z
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA; // matAtA为海塞矩阵，对应的特征值分别是roll,pitch,yaw,x,y,z方向的特征值
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            // 判断是否退化是在第一次迭代里面发生的，发布第一次的海塞矩阵到话题中
            publishHessianMatrix(matAtA);
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV); // 对应的特征值分别是roll,pitch,yaw,x,y,z方向的特征值
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true;  // converged
        }
        return false;  // keep optimizing
    }

    void scan2MapOptimization() {
        if (cloudKeyPoses3D->points.empty()) return;

        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum &&
            laserCloudSurfLastDSNum > surfFeatureMinValidNum) {
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 30; iterCount++) {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization();
                surfOptimization();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true) break;
            }

            transformUpdate();
        } else {
            ROS_WARN(
                    "Not enough features! Only %d edge and %d planar features available.",
                    laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    void transformUpdate() {
        if (cloudInfo.imuAvailable == true) {
            if (std::abs(cloudInfo.imuPitchInit) < 1.4) {
                double imuWeight = imuRPYWeight;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
                        .getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
                        .getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] =
                constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] =
                constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] =
                constraintTransformation(transformTobeMapped[5], z_tollerance);
        // scan2map的优化后，存储的结果
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit) {
        if (value < -limit) value = -limit;
        if (value > limit) value = limit;

        return value;
    }

    bool saveFrame() {
        if (cloudKeyPoses3D->points.empty()) return true;

        if (sensor == SensorType::LIVOX) {
            if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0) return true;
        }

        // 取出上一个关键帧的位姿
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        // 取出当前帧的位姿转换成eigen形式
        Eigen::Affine3f transFinal = pcl::getTransformation(
                transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // 计算上一个关键帧和当前关键帧之间的位姿差
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        // 把两个相对位姿变换转换成平移和欧拉角
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
            return false;

        // std::cout << "distance gap: " << sqrt(x * x + y * y) << std::endl;
        keyframeDistances.push_back(sqrt(x * x + y * y));

        return true;
    }

    void addOdomFactor() {
        if (cloudKeyPoses3D->points.empty()) {
            // 第一个关键帧的先验置信度
            // 如果使用真值提供的初始位姿，就给高的置信度
            noiseModel::Diagonal::shared_ptr priorNoise;
            if (uesInitialVehiclePose) {
                priorNoise =  noiseModel::Diagonal::Variances((Vector(6) << 1e-12, 1e-12, 1e-12, 1e-10, 1e-10, 1e-10).finished());
            } else {
                priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished());
            }
            // 增加先验约束，对第0个节点增加约束
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped),
                                              priorNoise));
            // 加入节点信息
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        } else {
            // 如果不是第一个关键帧，就增加帧间约束，此时帧间约束的置信度较高
            noiseModel::Diagonal::shared_ptr odometryNoise =
                    noiseModel::Diagonal::Variances(
                            (Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            // poseFrom为最后一个关键帧的位姿，poseTo为当前关键帧通过scantoMap计算的位姿
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);

            mtxGraph.lock();
            // 这是一个帧间约束，分别输入两个节点的id，帧间约束大小以及置信度
            gtSAMgraph.add(BetweenFactor<Pose3>(
                    cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(),
                    poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
            mtxGraph.unlock();
        }
    }

    void addRoadSideFactor() {
        if (roadRegistrationQueue.empty() || cloudKeyPoses3D->points.empty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mtxRoadSideInfo);
        auto roadRegMsg = roadRegistrationQueue.front();
        roadRegistrationQueue.pop_front();
    
        // 从消息中获取关键帧ID
        const int32_t msgKeyframeId = roadRegMsg.keyFrameId;
        if (msgKeyframeId < 0) {
            ROS_ERROR("Invalid keyframe ID: %d", msgKeyframeId);
            return;
        }
        int currKeyframeId = static_cast<int>(msgKeyframeId);
        if (debugRoadSide) {
            ROS_INFO_STREAM("[addRoadSideFactor] 2 currKeyframeId:" << currKeyframeId);
        }

        // Vehicle frame to roadside frame的变换矩阵
        Eigen::Isometry3d T_GNSSN_vehicle = Eigen::Isometry3d::Identity();
        T_GNSSN_vehicle.translation() << 
            roadRegMsg.pose.position.x,
            roadRegMsg.pose.position.y,
            roadRegMsg.pose.position.z;
        T_GNSSN_vehicle.linear() = Eigen::Quaterniond(
            roadRegMsg.pose.orientation.w,
            roadRegMsg.pose.orientation.x,
            roadRegMsg.pose.orientation.y,
            roadRegMsg.pose.orientation.z
        ).normalized().toRotationMatrix();

        Eigen::Isometry3d T_n_vehicle = Eigen::Isometry3d(T_GNSSN_vehicle) ;
        if (debugRoadSide) {
            std::cout << "T_n_vehicle =\n" << T_n_vehicle.matrix() << std::endl;
        }

        // 转换为GTSAM格式（保持原逻辑）
        const gtsam::Pose3 pose3(gtsam::Rot3(T_n_vehicle.linear()), gtsam::Point3(T_n_vehicle.translation()));
        // 固定噪声模型（保持原逻辑）
        static const gtsam::Vector6 fixedNoise = (gtsam::Vector6() << 0.1, 0.1, 0.1, 0.01, 0.01, 0.01).finished();
        static const auto noiseModel = gtsam::noiseModel::Diagonal::Variances(fixedNoise);
        // 添加先验因子（保持原逻辑）
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(currKeyframeId, pose3, noiseModel));

        // 存储索引和位姿点云（坐标系转换后的位置）  下面这句话会报错
        // roadIndexContainer[currKeyframeId] = static_cast<int>(cloudKeyroadSidePoses3D->points.size());

        PointType roadSidePoint;
        roadSidePoint.getVector3fMap() = T_n_vehicle.translation().cast<float>();
        if (debugRoadSide) {
            ROS_INFO_STREAM("[addRoadSideFactor] 7");
        }
        // cloudKeyroadSidePoses3D->points.push_back(roadSidePoint); 这句话会报错，但是不知道为什么
        aLoopIsClosed = true;
        if (debugRoadSide) {
            ROS_INFO_STREAM("[Add RoadFactor] for keyframe" << currKeyframeId);
        }
    }

    void addGPSFactor() {
        if (gpsQueue.empty()) return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty() || cloudKeyPoses3D->points.size() == 1)
            return;
        //    else {
        //      if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back())
        //      < 5.0)
        //        return;
        //    }

        // pose covariance small, no need to correct
        //        if (poseCovariance(3, 3) < poseCovThreshold && poseCovariance(4,
        //        4) < poseCovThreshold)
        //            return;

        // last gps position
        static PointType lastGPSPoint;
        nav_msgs::Odometry thisGPS;
        if (syncGPS(gpsQueue, thisGPS, timeLaserInfoCur, 1.0 / gpsFrequence)) {
            // GPS too noisy, skip
            float noise_x = thisGPS.pose.covariance[0];
            float noise_y = thisGPS.pose.covariance[7];
            float noise_z = thisGPS.pose.covariance[14];

            // make sure the gps data is stable encough
            if (abs(noise_x) > gpsCovThreshold || abs(noise_y) > gpsCovThreshold)
                return;

//            float gps_x = thisGPS.pose.pose.position.x;
//            float gps_y = thisGPS.pose.pose.position.y;
//            float gps_z = thisGPS.pose.pose.position.z;
            double gps_x = 0.0, gps_y = 0.0, gps_z = 0.0;
            Eigen::Vector3d LLA(thisGPS.pose.covariance[1], thisGPS.pose.covariance[2], thisGPS.pose.covariance[3]);
            geo_converter.Forward(LLA[0], LLA[1], LLA[2], gps_x, gps_y, gps_z);
            // 考虑是否使用GPS的高度信息
            if (!useGpsElevation) {
                gps_z = transformTobeMapped[5];
                noise_z = 0.01;
            }
            // GPS not properly initialized (0,0,0)
            if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6) return;

            // Add GPS every a few meters
            PointType curGPSPoint;
            curGPSPoint.x = gps_x;
            curGPSPoint.y = gps_y;
            curGPSPoint.z = gps_z;
            if (pointDistance(curGPSPoint, lastGPSPoint) < GPSDISTANCE)
                return;
            else
                lastGPSPoint = curGPSPoint;

            if (debugGps) {
                ROS_INFO("curr gps pose: %f, %f , %f", gps_x, gps_y, gps_z);
                ROS_INFO("curr gps cov: %f, %f , %f", thisGPS.pose.covariance[0],
                         thisGPS.pose.covariance[7], thisGPS.pose.covariance[14]);
            }

            gtsam::Vector Vector3(3);
            Vector3 << noise_x, noise_y, noise_z;
            // Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
            noiseModel::Diagonal::shared_ptr gps_noise =
                    noiseModel::Diagonal::Variances(Vector3);
            gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(),
                                        gtsam::Point3(gps_x, gps_y, gps_z),
                                        gps_noise);
            keyframeGPSfactor.push_back(gps_factor);
            cloudKeyGPSPoses3D->points.push_back(curGPSPoint);

            // only a trick!
            // we need to accumulate some accurate gps points to initialize the
            // transform between gps coordinate system and LIO coordinate system and
            // then we can add gps points one by one into the pose graph or the whole
            // pose graph will crashed if giving some respectively bad gps points at
            // first.
            if (keyframeGPSfactor.size() < 20) {
                ROS_INFO("Accumulated gps factor: %d", keyframeGPSfactor.size());
                return;
            }

            if (!gpsTransfromInit) {
                ROS_INFO("Initialize GNSS transform!");
                for (int i = 0; i < keyframeGPSfactor.size(); ++i) {
                    gtsam::GPSFactor gpsFactor = keyframeGPSfactor.at(i);
                    gtSAMgraph.add(gpsFactor);
                    gpsIndexContainer[gpsFactor.key()] = i;
                }
                gpsTransfromInit = true;
            } else {
                // After the coordinate systems are aligned, in theory, the GPS z and
                // the z estimated by the current LIO system should not be too
                // different. Otherwise, there is a problem with the quality of the
                // secondary GPS point.
                //                if (abs(gps_z - cloudKeyPoses3D->back().z) > 10.0) {
                //                    // ROS_WARN("Too large GNSS z noise %f", noise_z);
                //                    gtsam::Vector Vector3(3);
                //                    Vector3 << max(noise_x, 10000.0f), max(noise_y,
                //                    10000.0f), max(noise_z, 100000.0f);
                //                    // gps_noise =
                //                    noiseModel::Diagonal::Variances(Vector3);
                //                    // gps_factor =
                //                    gtsam::GPSFactor(cloudKeyPoses3D->size(),
                //                    gtsam::Point3(gps_x, gps_y, gps_z),
                //                    // gps_noise);
                //                }
                // add loop constriant
                mtxGraph.lock();
                gtSAMgraph.add(gps_factor);
                mtxGraph.unlock();
                gpsIndexContainer[cloudKeyPoses3D->size()] =
                        cloudKeyGPSPoses3D->size() - 1;
            }
            // 加入gps之后等同于回环，需要触发较多的isam update
            aLoopIsClosed = true;
        }
    }

    void addLoopFactor() {
        // 有一个专门的回环检测线程会检测回环，检测到就会给这个队列塞入回环结果
        if (loopIndexQueue.empty()) return;

        for (int i = 0; i < (int) loopIndexQueue.size(); ++i) {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            mtxGraph.lock();
            gtSAMgraph.add(
                    BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
            mtxGraph.unlock();
        }

        // 清空回环相关的队列
        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        // 回环的标志位置为true
        aLoopIsClosed = true;
    }

    void saveKeyFramesAndFactor() {
        if (saveFrame() == false) return;

        // odom factor
        addOdomFactor();

        // gps factor
        if (useGPS) addGPSFactor();

        if (useRoadSide) {
            addRoadSideFactor();
        }

        // loop factor
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // add raw odom
        nav_msgs::Odometry laserOdometryROS;
        transformEiegn2Odom(timeLaserInfoCur, laserOdometryROS,
                            transformTobeMapped);
        keyframeRawOdom.push_back(laserOdometryROS);

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        // 如果加入了gps约束或者回环约束，就需要多次更新isam
        if (aLoopIsClosed == true) {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }
        // 将约束和节点信息清空，他们已经加入到isam中了，因此清空不会影响整个优化
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        // save key poses 接下来保存关键帧信息
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate(); // 优化后的所有关键帧的位姿都存在这里面
        // 当前关键帧优化后的位姿
        latestEstimate =
                isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity =
                cloudKeyPoses3D->size();  // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity;  // this can be used as index
        thisPose6D.roll = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl <<
        // endl;
        // 保存优化后当前位姿的置信度
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);
        // save updated transform  将优化后的位姿保存到transformTobeMapped中，作为当前最佳估计值
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(
                new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(
                new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thislaserCloudRawKeyFrame(
                new pcl::PointCloud<PointType>());
        // 当前帧的点云的角点和面点分别拷贝一下
        pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);
//        pcl::copyPointCloud(*laserCloudRawDS, *thislaserCloudRawKeyFrame);
        pcl::copyPointCloud(*laserCloudRaw, *thislaserCloudRawKeyFrame);

        // save key frame cloud  保存关键帧的点云
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
        // if you want to save raw cloud
        laserCloudRawKeyFrames.push_back(thislaserCloudRawKeyFrame); // 保存了原始点云，但是是当前帧的点云
        keyframeCloudDeskewed.push_back(cloudInfo.cloud_deskewed);
        keyframeTimes.push_back(timeLaserInfoStamp.toSec());

        // save keyframe pose odom
        //        nav_msgs::Odometry updatesOdometryROS;
        //        transformEiegn2Odom(timeLaserInfoCur, updatesOdometryROS,
        //        transformTobeMapped);
        //        keyframePosesOdom.push_back(updatesOdometryROS);

        // save path for visualization  根据当前最新位姿更新rviz可视化
        updatePath(thisPose6D);
    }

    void correctPoses() {
        // 没有关键帧，自然也没有意义
        if (cloudKeyPoses3D->points.empty()) return;
        // 只有当回环以及gps信息这些会触发，全局调整才会被触发，因为此时对全局的位姿影响比较大，因此需要将之前的位姿全部清空并再重新赋值
        if (aLoopIsClosed == true) {
            // clear map cache 这里面存储关键帧位姿和关键帧点云
            laserCloudMapContainer.clear();
            // clear path  清空path
            globalPath.poses.clear();

            // update key poses  更新所有的位姿
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i) {
                // 更新所有关键帧的位姿
                cloudKeyPoses3D->points[i].x =
                        isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y =
                        isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z =
                        isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll =
                        isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch =
                        isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw =
                        isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }
            // 全局位姿已经调整完成，然后将回环标志位置为false
            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose &pose_in) {
        geometry_msgs::PoseStamped pose_stamped;

        /*----------------------------------------------------------*/
        // transform Current frame to Body_imu frame
        //    tf::Quaternion rot_to_bodyimu(0.003148, -0.002479, 0.000524,0.999992);
        //    //tf::Quaternion rot_to_bodyimu(0,0,0,1);
        //    tf::Vector3 trans_to_bodyimu(-0.047781, 0.007303, -0.026583);
        //    tf::Transform current_to_bodyimu(rot_to_bodyimu,trans_to_bodyimu);
        //    tf::Transform current_pose(tf::createQuaternionFromRPY(pose_in.roll,
        //    pose_in.pitch, pose_in.yaw),
        //                               tf::Vector3(pose_in.x, pose_in.y,
        //                               pose_in.z));
        //    //std::cout<< "previous pose: " << current_pose.getOrigin().x() <<"
        //    "<<current_pose.getOrigin().y()<<"  " <<current_pose.getOrigin().z()
        //    << std::endl; current_pose = current_to_bodyimu * current_pose;
        // std::cout<< "Transformed pose: " << current_pose.getOrigin().x()<<"  "
        // <<current_pose.getOrigin().y()<<"  " <<current_pose.getOrigin().z() <<
        // "\n" << std::endl;

        //    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        //    pose_stamped.header.frame_id = odometryFrame;
        //    pose_stamped.pose.position.x = current_pose.getOrigin().x();
        //    pose_stamped.pose.position.y = current_pose.getOrigin().y();
        //    pose_stamped.pose.position.z = current_pose.getOrigin().z();
        //    pose_stamped.pose.orientation.x =  current_pose.getRotation().x();
        //    pose_stamped.pose.orientation.y =  current_pose.getRotation().y();
        //    pose_stamped.pose.orientation.z =  current_pose.getRotation().z();
        //    pose_stamped.pose.orientation.w =  current_pose.getRotation().w();
        /*----------------------------------------------------------*/

        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q =
                tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }


    void PrintTransformTobeMapped(const std::string &functionName) {
        if (debugInitialVehiclePose) {
            std::lock_guard<std::mutex> lock(debugKeyFrameMutex);
            std::cout << functionName << " transformTobeMapped: ";
            for (int i = 0; i < 6; ++i) {
                std::cout << std::fixed << std::setprecision(3) << transformTobeMapped[i] << " ";
            }
            std::cout << std::endl;
        }
    }

    void publishOdometry() {
        // Publish odometry for ROS (global)
        nav_msgs::Odometry laserOdometryROS;
        transformEiegn2Odom(timeLaserInfoCur, laserOdometryROS,
                            transformTobeMapped);
        //        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        //        laserOdometryROS.header.frame_id = odometryFrame;
        //        laserOdometryROS.child_frame_id = "odom_mapping";
        //        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        //        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        //        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        //        laserOdometryROS.pose.pose.orientation =
        //                tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0],
        //                transformTobeMapped[1],
        //                                                        transformTobeMapped[2]);
        pubLaserOdometryGlobal.publish(laserOdometryROS);

        // Publish TF
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(
                tf::createQuaternionFromRPY(transformTobeMapped[0],
                                            transformTobeMapped[1],
                                            transformTobeMapped[2]),
                tf::Vector3(transformTobeMapped[3], transformTobeMapped[4],
                            transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(
                t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        br.sendTransform(trans_odom_to_lidar);

        if (useGPS) {
//            if (gpsTransfromInit) {
            /** we first update the initial GPS origin points since it may not fix here */
            //                Eigen::Vector3d origin_point(cloudKeyPoses6D->at(0).x,
            //                                             cloudKeyPoses6D->at(0).y,
            //                                             cloudKeyPoses6D->at(0).z);
            //                // ENU->LLA
            //                Eigen::Vector3d update_origin_lla;
            //                geo_converter.Reverse(origin_point[0], origin_point[1], origin_point[2], update_origin_lla[0],
            //                                      update_origin_lla[1],
            //                                      update_origin_lla[2]);
            //geo_converter.Reset(update_origin_lla[0], update_origin_lla[1], update_origin_lla[2]);
            // std::cout << " origin points: " << originLLA.transpose() << std::endl;
            // std::cout << " update origin points: " << update_origin_lla.transpose() << std::endl;
            //                originLLA = update_origin_lla;
            //                updateOrigin = false;
            // ROS_WARN("UPDATE MAP ORIGIN SUCCESS!");
//            }

            /** we transform the  ENU point to LLA point for visualization with rviz_satellite*/
            Eigen::Vector3d curr_point(cloudKeyPoses6D->back().x,
                                       cloudKeyPoses6D->back().y,
                                       cloudKeyPoses6D->back().z);
            Eigen::Vector3d curr_lla;
            // ENU->LLA
            geo_converter.Reverse(curr_point[0], curr_point[1], curr_point[2], curr_lla[0], curr_lla[1],
                                  curr_lla[2]);
            //                std::cout << std::setprecision(9)
            //                          << "CURR LLA: " << originLLA.transpose() << std::endl;
            //                std::cout << std::setprecision(9)
            //                          << "update LLA: " << curr_lla.transpose() << std::endl;
            sensor_msgs::NavSatFix fix_msgs;
            fix_msgs.header.stamp = ros::Time().fromSec(timeLaserInfoCur);
            fix_msgs.header.frame_id = odometryFrame;
            fix_msgs.latitude = curr_lla[0];
            fix_msgs.longitude = curr_lla[1];
            fix_msgs.altitude = curr_lla[2];
            pubGPSOdometry.publish(fix_msgs);
        }

        // Publish odometry for ROS (incremental)
        // 发送增量的位姿变换信息，主要是给imu预积分模块使用的，需要里程计是足够的平滑
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental;  // incremental odometry msg
        static Eigen::Affine3f increOdomAffine;  // incremental odometry in affine
        if (lastIncreOdomPubFlag == false) {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            // 上一帧最佳位姿和当前帧最佳位姿(当前scan2map后的位姿，而不是gps和回环调整后的位姿)之间的位姿增量
            // 也就是scan2map前后的位姿结果
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() *
                                          incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre; // 这个是发布给imu的话题内容
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles(increOdomAffine, x, y, z, roll, pitch,
                                              yaw);
            if (cloudInfo.imuAvailable == true) {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4) {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
                            .getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
                            .getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation =
                    tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        // 回环的信息并不会体现在这里面，只有增量的信息在里面，也就是只发布scan2map的增量信息到imu预积分模块
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    void publishFrames() {
        if (cloudKeyPoses3D->points.empty()) return;
        // publish key poses
        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp,
                     odometryFrame);
        // Publish surrounding key frames
        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS,
                     timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame.getNumSubscribers() != 0) {
            pcl::PointCloud<PointType>::Ptr cloudOut(
                    new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS, &thisPose6D);
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp,
                         odometryFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0) {
            pcl::PointCloud<PointType>::Ptr cloudOut(
                    new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut, &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp,
                         odometryFrame);
        }
        if (pubCloudRaw.getNumSubscribers() != 0) {
            pcl::PointCloud<PointType>::Ptr cloudOut(
                    new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            publishCloud(pubCloudRaw, cloudOut, timeLaserInfoStamp, lidarFrame);
        }

        // publish path
        if (pubPath.getNumSubscribers() != 0) {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
        // publish SLAM infomation for 3rd-party usage，这里拿来给路侧lidar使用
        static int lastSLAMInfoPubSize = -1;
        if (pubSLAMInfo.getNumSubscribers() != 0) {
            if (lastSLAMInfoPubSize != cloudKeyPoses6D->size()) {
                lio_sam_6axis::cloud_info slamInfo;
                slamInfo.header.stamp = timeLaserInfoStamp;
                pcl::PointCloud<PointType>::Ptr cloudOut(
                        new pcl::PointCloud<PointType>());
                *cloudOut += *laserCloudCornerLastDS;
                *cloudOut += *laserCloudSurfLastDS;
                slamInfo.key_frame_cloud = publishCloud(ros::Publisher(), cloudOut,
                                                        timeLaserInfoStamp, lidarFrame);
                slamInfo.key_frame_poses =
                        publishCloud(ros::Publisher(), cloudKeyPoses6D, timeLaserInfoStamp,
                                     odometryFrame); // 这里面发布的是所有的关键帧6D的位姿以及时间戳和索引
                pcl::PointCloud<PointType>::Ptr localMapOut(
                        new pcl::PointCloud<PointType>());
                *localMapOut += *laserCloudCornerFromMapDS;
                *localMapOut += *laserCloudSurfFromMapDS;
                slamInfo.key_frame_map = publishCloud(
                        ros::Publisher(), localMapOut, timeLaserInfoStamp, odometryFrame);
                // 新增发布降采样后的关键帧点云，这个点云已经去畸变了，路侧lidar可以使用
                slamInfo.cloud_deskewed = publishCloud(ros::Publisher(), laserCloudRawDS,
                                                       timeLaserInfoStamp, lidarFrame);
                pubSLAMInfo.publish(slamInfo);
                lastSLAMInfoPubSize = cloudKeyPoses6D->size();
            }
        }
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "lio_sam_6axis");
    mapOptimization MO;
    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread,
                                   &MO);

    ros::spin();

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
