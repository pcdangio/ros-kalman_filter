/// \file kalman_filter/ukf/model.hpp
/// \brief Defines the kalman_filter::ukf::model_t class.
#ifndef KALMAN_FILTER___UKF___MODEL_H
#define KALMAN_FILTER___UKF___MODEL_H

#include <kalman_filter/ukf/observer.hpp>

#include <unordered_map>
#include <memory>

namespace kalman_filter {
namespace ukf {

/// \brief A base-class definition of a nonlinear system state space model.
struct model_t
{
    // CONSTRUCTORS
    /// \brief Instantiates a new model.
    /// \param dimensions The number of dimensions in the model's state.
    model_t(uint32_t dimensions);

    // METHODS
    /// \brief Calculates a transition from one state to a new state.
    /// \param xp The current state to transition from.
    /// \param q The process noise vector.
    /// \param x The calculated new state.
    virtual void state_transition(const Eigen::VectorXd& xp, const Eigen::VectorXd& q, Eigen::VectorXd& x) = 0;

    // VARIABLES
    /// \brief The number of dimensions in the model's state.
    const uint32_t dimensions;
    /// \brief The covariance of the model's noise.
    Eigen::MatrixXd Q;
    /// \brief The model's observers.
    std::unordered_map<observer_id_t, std::shared_ptr<observer_t>> observers;
};

}}

#endif