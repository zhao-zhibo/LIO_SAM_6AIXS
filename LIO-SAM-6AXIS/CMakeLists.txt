cmake_minimum_required(VERSION 2.8.3)
project(lio_sam_6axis)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")
find_package(Boost REQUIRED COMPONENTS timer thread date_time)
find_package(catkin REQUIRED COMPONENTS
        tf
        roscpp
        rospy
        cv_bridge
        pcl_conversions
        std_msgs
        sensor_msgs
        geometry_msgs
        nav_msgs
        message_generation
        visualization_msgs
        rosbag
        )

find_package(OpenMP REQUIRED)
# find_package(PCL REQUIRED)
find_package(PCL 1.12 REQUIRED COMPONENTS common io visualization filters features kdtree registration)
message(STATUS "Found PCL version: ${PCL_VERSION}")
if(PCL_VERSION VERSION_LESS 1.12)
  message(FATAL_ERROR "PCL 1.12+ required, found ${PCL_VERSION}")
endif()

find_package(OpenCV REQUIRED)
find_package(GTSAM REQUIRED)
find_package(libxml2 QUIET)
# # 显式链接VTK所有组件（补充缺失模块）
# find_package(VTK REQUIRED COMPONENTS
#   vtkCommonCore
#   vtkRenderingCore
#   vtkInteractionStyle
# )
if (NOT LIBXML2_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(Y REQUIRED libxml-2.0)
endif ()
message(STATUS "PCL_LIBRARIES: ${PCL_LIBRARIES}")
message(STATUS "PCL_INCLUDE_DIRS: ${PCL_INCLUDE_DIRS}")
message(STATUS "GTSAM_INCLUDE_DIR: ${GTSAM_INCLUDE_DIR}")

add_message_files(
        DIRECTORY msg
        FILES
        cloud_info.msg
        road_registration.msg
)

add_service_files(
        DIRECTORY srv
        FILES
        save_map.srv
        transformInterp.srv
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
        std_msgs
        nav_msgs
        sensor_msgs
)

add_subdirectory(./ThirdParty/GeographicLib/)
add_subdirectory(./ThirdParty/small_gicp/)

catkin_package(
        INCLUDE_DIRS include
        DEPENDS PCL GTSAM

        CATKIN_DEPENDS
        std_msgs
        nav_msgs
        geometry_msgs
        sensor_msgs
        message_runtime
        message_generation
        visualization_msgs
)

# 安装geoFunc模块 [!++]
install(
  DIRECTORY scripts/geoFunc/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/geoFunc
)

# include directories
include_directories(
        include
        /usr/local/include                    # [!新增!]
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIR}
        ${Y_INCLUDE_DIRS}
        ${LIBXML2_INCLUDE_DIRS}
        # ${VTK_INCLUDE_DIRS}
        ./ThirdParty/GeographicLib/include/
        ./ThirdParty/small_gicp/include/
        ./ThirdParty/small_gicp/include/ann
        ./ThirdParty/small_gicp/include/benchmark
        ./ThirdParty/small_gicp/include/factors
        ./ThirdParty/small_gicp/include/pcl
        ./ThirdParty/small_gicp/include/points
        ./ThirdParty/small_gicp/include/registration
        ./ThirdParty/small_gicp/include/util
)

# include(${VTK_USE_FILE})

# link directories
link_directories(
        include
        ${PCL_LIBRARY_DIRS}
        ${OpenCV_LIBRARY_DIRS}
        ${GTSAM_LIBRARY_DIRS}
        ${LIBXML2_LIBRARY_DIR}
        ${Y_LIBRARY_DIRS}
        # ${VTK_LIBRARY_DIRS}
)

###########
## Build ##
###########

# Gps odometry
add_executable(${PROJECT_NAME}_gpsOdometry src/simpleGpsOdom.cpp)
target_link_libraries(${PROJECT_NAME}_gpsOdometry ${catkin_LIBRARIES} ${LIBXML2_LIBRARIES} ${Y_LIBRARIES} libGeographic)

# Range Image Projection
add_executable(${PROJECT_NAME}_imageProjection src/imageProjection.cpp)
add_dependencies(${PROJECT_NAME}_imageProjection ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${LIBXML2_LIBRARIES} ${Y_LIBRARIES})

# Feature Association
add_executable(${PROJECT_NAME}_featureExtraction src/featureExtraction.cpp)
add_dependencies(${PROJECT_NAME}_featureExtraction ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_featureExtraction ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${LIBXML2_LIBRARIES} ${Y_LIBRARIES})

# Mapping Optimization
add_executable(${PROJECT_NAME}_mapOptmization src/mapOptmizationGps.cpp src/dataSaver.cpp)
add_dependencies(${PROJECT_NAME}_mapOptmization ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_mapOptmization PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_mapOptmization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam ${LIBXML2_LIBRARIES} ${Y_LIBRARIES} libGeographic)

# IMU Preintegration
add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
target_link_libraries(${PROJECT_NAME}_imuPreintegration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam ${LIBXML2_LIBRARIES} ${Y_LIBRARIES})

# RoadSide Process
add_executable(${PROJECT_NAME}_roadSide src/roadSide.cpp)
add_dependencies(${PROJECT_NAME}_roadSide ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_roadSide  SmallGicp
        ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam ${LIBXML2_LIBRARIES} ${Y_LIBRARIES} libGeographic )

# 插入python文件的执行
catkin_install_python(PROGRAMS scripts/callAllCommand.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

# 插入python文件的执行
catkin_install_python(PROGRAMS scripts/transformServiceNode.py  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

# 插入python文件的执行
catkin_install_python(PROGRAMS scripts/plotHessian.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})