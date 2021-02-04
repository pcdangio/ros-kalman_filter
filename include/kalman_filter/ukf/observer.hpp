/// \file kalman_filter/ukf/observer.hpp
/// \brief Defines the kalman_filter::ukf::observer_t class.
#ifndef KALMAN_FILTER___UKF___OBSERVER_H
#define KALMAN_FILTER___UKF___OBSERVER_H

#include <eigen3/Eigen/Dense>

namespace kalman_filter {
namespace ukf {

/// \brief A type definition for a unique observer ID.
typedef uint32_t observer_id_t;

/// \brief A base-class descriptor for a model state observer.
struct observer_t
{
    // CONSTRUCTORS
    /// \brief Instantiates a new observer.
    /// \param id The unique ID of the observer.
    /// \param dimensions The number of dimensions in each observation.
    observer_t(observer_id_t id, uint32_t dimensions);

    // METHODS
    /// \brief Calculates the observation for a given model state.
    /// \param x The state to calculate the observation for.
    /// \param r The observation noise vector.
    /// \param z The calculated observation.
    virtual void observation_model(const Eigen::VectorXd& x, const Eigen::VectorXd& r, Eigen::VectorXd& z) = 0;

    // VARIABLES
    /// \brief The observer's unique ID.
    const observer_id_t id;
    /// \brief The number of dimensions in each observation.
    const uint32_t dimensions;
    /// \brief The covariance of observeration noise.
    Eigen::MatrixXd R;
};

}}

#endif