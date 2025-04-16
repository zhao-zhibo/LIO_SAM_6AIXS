// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>

namespace small_gicp {

/// @brief Registration termination criteria
struct TerminationCriteria {
  /// @brief Constructor
  // 现在设置的收敛阈值，translation_eps(对应optimizer.hpp中的delta中的dt)平移量是0.001米，rotation_eps(对应optimizer.hpp中的delta中的dr)是旋转量是0.1度
  // 进行路侧lidar和车载lidar的配准时，平移的阈值一直无法收敛，但是旋转的阈值可以收敛，因此我将平移的阈值增大
  TerminationCriteria() : translation_eps(1e-2), rotation_eps(0.1 * M_PI / 180.0) {}

  /// @brief Check the convergence
  /// @param delta  Transformation update vector, 前三个量head是旋转，后三个量tail是平移
  /// @return       True if converged
  bool converged(const Eigen::Matrix<double, 6, 1>& delta) const { return delta.template head<3>().norm() <= rotation_eps && delta.template tail<3>().norm() <= translation_eps; }

  double translation_eps;  ///< Rotation tolerance [rad]
  double rotation_eps;     ///< Translation tolerance
};

}  // namespace small_gicp
