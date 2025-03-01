project (SmallGicpLib)

# 设置项目版本信息
set (PROJECT_VERSION_MAJOR 1)
set (PROJECT_VERSION_MINOR 0)
set (PROJECT_VERSION_PATCH 0)
set (PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")
if (PROJECT_VERSION_PATCH GREATER 0)
  set (PROJECT_VERSION "${PROJECT_VERSION}.${PROJECT_VERSION_PATCH}")
endif ()

# 设置CMake的最低版本要求
cmake_minimum_required (VERSION 3.0) # 根据需要调整版本

# 设置库名称
set (LIBNAME SmallGicp)

# 包含头文件目录
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/ann
  ${CMAKE_CURRENT_SOURCE_DIR}/include/benchmark
  ${CMAKE_CURRENT_SOURCE_DIR}/include/factors
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pcl
  ${CMAKE_CURRENT_SOURCE_DIR}/include/points
  ${CMAKE_CURRENT_SOURCE_DIR}/include/registration
  ${CMAKE_CURRENT_SOURCE_DIR}/include/util
)

# 由于只有hpp文件，不需要添加库文件
# add_library(${LIBNAME} ...)

# 安装头文件
install(DIRECTORY include/ DESTINATION include)