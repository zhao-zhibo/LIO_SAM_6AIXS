// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>

namespace small_gicp {

/// @brief Registration termination criteria
struct TerminationCriteria {
  /// @brief Constructor
  // 现在设置的收敛阈值，平移量是0.001米，旋转量是0.2度，之前旋转量是0.1度
  TerminationCriteria() : translation_eps(1e-3), rotation_eps(0.2 * M_PI / 180.0) {}

  /// @brief Check the convergence
  /// @param delta  Transformation update vector
  /// @return       True if converged
  bool converged(const Eigen::Matrix<double, 6, 1>& delta) const { return delta.template head<3>().norm() <= rotation_eps && delta.template tail<3>().norm() <= translation_eps; }

  double translation_eps;  ///< Rotation tolerance [rad]
  double rotation_eps;     ///< Translation tolerance
};

}  // namespace small_gicp
