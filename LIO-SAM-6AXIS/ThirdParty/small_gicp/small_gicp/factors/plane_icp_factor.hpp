// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <small_gicp/util/lie.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/points/traits.hpp>

namespace small_gicp {

/// @brief Point-to-plane per-point error factor.
struct PointToPlaneICPFactor {
  struct Setting {};

  PointToPlaneICPFactor(const Setting& setting = Setting()) : target_index(std::numeric_limits<size_t>::max()), source_index(std::numeric_limits<size_t>::max()) {}

  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree, typename CorrespondenceRejector>
  bool linearize(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const Eigen::Isometry3d& T,
    size_t source_index,
    const CorrespondenceRejector& rejector,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* e) {

    // 步骤1：初始化索引
    this->source_index = source_index;
    this->target_index = std::numeric_limits<size_t>::max();

    // 步骤2：坐标变换
    const Eigen::Vector4d transed_source_pt = T * traits::point(source, source_index);

    // 步骤3：最近邻搜索
    size_t k_index;
    double k_sq_dist;
    if (!traits::nearest_neighbor_search(target_tree, transed_source_pt, &k_index, &k_sq_dist) || rejector(target, source, T, k_index, source_index, k_sq_dist)) {
      return false; // 对应关系被拒绝
    }

    // 步骤4：获取法向量
    target_index = k_index;
    const auto& target_normal = traits::normal(target, target_index);

    // 步骤5：计算残差
    const Eigen::Vector4d residual = traits::point(target, target_index) - transed_source_pt;
    const Eigen::Vector4d err = target_normal.array() * residual.array(); // 点对面投影

    // 步骤6：计算雅可比矩阵
    Eigen::Matrix<double, 4, 6> J = Eigen::Matrix<double, 4, 6>::Zero();
    // 旋转部分导数：-n^T * [T.R * (p_s ×)] 
    J.block<3, 3>(0, 0) = target_normal.template head<3>().asDiagonal() * T.linear() * skew(traits::point(source, source_index).template head<3>());
    // 平移部分导数：-n^T * T.R
    J.block<3, 3>(0, 3) = target_normal.template head<3>().asDiagonal() * (-T.linear());

    // Eigen::Matrix<double, 1, 6> J = Eigen::Matrix<double, 1, 6>::Zero();
    // const Eigen::Vector3d p_source = traits::point(source, source_index).template head<3>();
    // J.block<1, 3>(0, 0) = -target_normal.template head<3>().transpose() * T.linear() * skew(p_source);
    // J.block<1, 3>(0, 3) = -target_normal.template head<3>().transpose() * T.linear();

    // 步骤7：构建Hessian和梯度
    *H = J.transpose() * J;  // H = J^T J
    *b = J.transpose() * err;  // b = J^T e
    *e = 0.5 * err.squaredNorm();  // 误差项

    return true;
  }

  template <typename TargetPointCloud, typename SourcePointCloud>
  double error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T) const {
    if (target_index == std::numeric_limits<size_t>::max()) {
      return 0.0;
    }

    const Eigen::Vector4d transed_source_pt = T * traits::point(source, source_index);
    const Eigen::Vector4d residual = traits::point(target, target_index) - transed_source_pt;
    const Eigen::Vector4d error = traits::normal(target, target_index).array() * residual.array();
    return 0.5 * error.squaredNorm();
  }

  bool inlier() const { return target_index != std::numeric_limits<size_t>::max(); }

  size_t target_index;
  size_t source_index;
};
}  // namespace small_gicp
