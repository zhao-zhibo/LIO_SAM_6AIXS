cmake_minimum_required(VERSION 3.12)  # 需要支持INTERFACE库和现代语法
project(SmallGicp VERSION 1.0.0)

# 设置编译类型（默认为Release）
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_INSTALL_PREFIX "/usr/local")  # [!新增!]

# 查找OpenMP
find_package(OpenMP REQUIRED)

# 创建接口库
add_library(SmallGicp INTERFACE)

# 添加包含路径（构建时和安装后）
target_include_directories(SmallGicp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

# 链接OpenMP
target_link_libraries(SmallGicp INTERFACE OpenMP::OpenMP_CXX)

# 安装头文件（保留目录结构）
install(
  DIRECTORY small_gicp                  # [!修改!]
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
)