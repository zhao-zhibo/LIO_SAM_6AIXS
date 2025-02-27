#include "utility.h"
#include "lio_sam_6axis/cloud_info.h"
#include <benchmark/read_points.hpp>
#include <ann/kdtree_omp.hpp>
#include <points/point_cloud.hpp>
#include <factors/gicp_factor.hpp>
#include <factors/plane_icp_factor.hpp>
#include <util/downsampling_omp.hpp>
#include <util/normal_estimation_omp.hpp>
#include <registration/reduction_omp.hpp>
#include <registration/registration.hpp>

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

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;
class RoadSide : public ParamServer {
private:
    ros::Subscriber subRoadSideCloud;
    ros::Subscriber subVehicleCloud;

public:
    RoadSide() {

    // 订阅车载点云话题
    subVehicleCloud = nh.subscribe<lio_sam_6axis::cloud_info>("lio_sam_6axis/feature/cloud_info", 10, &RoadSide::VehicleCloudHandler, this,
            ros::TransportHints().tcpNoDelay());
    // 订阅路侧点云话题
    subRoadSideCloud = nh.subscribe<sensor_msgs::PointCloud2>(roadSidePointTopic, 10, &RoadSide::RoadSideCloudHandler, this,
            ros::TransportHints().tcpNoDelay()); }
    // 订阅车辆的位置和姿态信息

    // 发布路侧配准的结果

    void RoadSideCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg) {
    }

    void VehicleCloudHandler(const lio_sam_6axis::cloud_infoConstPtr &msgIn) {
    }

    void CloudRegistration(const std::vector<Eigen::Vector4f>& target_points, const std::vector<Eigen::Vector4f>& source_points) {
    }

};



int main(int argc, char **argv) {
    ros::init(argc, argv, "lio_sam_6axis");

    RoadSide RS;

    ROS_INFO("\033[1;32m----> RoadSide LiDAR Started.\033[0m");

    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();

    return 0;
}
