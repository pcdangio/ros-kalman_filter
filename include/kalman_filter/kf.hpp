/// \file kalman_filter/kf.hpp
/// \brief Defines the kalman_filter::kf_t class.
#ifndef KALMAN_FILTER___KF_H
#define KALMAN_FILTER___KF_H

#include <kalman_filter/base.hpp>

namespace kalman_filter {

/// \brief A Kalman Filter (KF)
/// \details The KF can perform linear state estimation with additive noise.
class kf_t
    : public base_t
{
public:
    // CONSTRUCTORS
    /// \brief Instantiates a new kf_t object.
    /// \param n_variables The number of variables in the state vector.
    /// \param n_inputs The number of inputs in the state model.
    /// \param n_observers The number of state observers.
    kf_t(uint32_t n_variables, uint32_t n_inputs, uint32_t n_observers);

    // FILTER METHODS
    void iterate() override;
    /// \brief Updates an input in the control input model.
    void new_input(uint32_t input_index, double_t input);

    // MODEL
    /// \brief The state transition model matrix.
    Eigen::MatrixXd A;
    /// \brief The control input model matrix.
    Eigen::MatrixXd B;
    /// \brief The observation model matrix.
    Eigen::MatrixXd H;

    // ACCESS
    /// \brief Gets the number of inputs in the state model.
    uint32_t n_inputs() const;

private:
    // DIMENSIONS
    /// \brief The number of inputs in the state model.
    uint32_t n_u;

    // STORAGE: PREDICT/UPDATE
    /// \brief The input vector.
    Eigen::VectorXd u;

    // STORAGE: TEMPORARIES
    /// \brief A temporary vector of size n_x.
    Eigen::VectorXd t_x;
    /// \brief A temporary matrix of size n_z,n_x.
    Eigen::MatrixXd t_zx;

    // Hide base class protected members.
    // NOTE: State variable and covariance access is still protected.
    using base_t::n_x;
    using base_t::n_z;
    using base_t::z;
    using base_t::S;
    using base_t::C;
    using base_t::t_xx;
    using base_t::has_observations;
    using base_t::masked_kalman_update;
};

}

#endif