/// \file kalman_filter/base.hpp
/// \brief Defines the kalman_filter::base_t class.
#ifndef KALMAN_FILTER___BASE_H
#define KALMAN_FILTER___BASE_H

#include <eigen3/Eigen/Dense>

#include <map>

/// \brief Contains objects for Kalman Filtering.
namespace kalman_filter {

/// \brief Provides base functionality for all Kalman Filter object types.
class base_t
{
public:
    // CONSTRUCTORS
    /// \brief Instantiates a new base_t object.
    /// \param n_variables The number of variables in the state vector.
    /// \param n_observers The number of state observers.
    base_t(uint32_t n_variables, uint32_t n_observers);

    // FILTER METHODS
    /// \brief Initializes the UKF with a specified state and covariance.
    /// \param initial_state The initial state vector to initialize with.
    /// \param initial_covariance The initial state covariance to initialize with.
    void initialize_state(const Eigen::VectorXd& initial_state, const Eigen::MatrixXd& initial_covariance);
    /// \brief Predicts a new state and performs update corrections with available observations.
    /// \note The iteration rate should be at least as fast as the fastest observer rate.
    virtual void iterate() = 0;
    /// \brief Adds a new observation to the filter.
    /// \param observer_index The index of the observer that made the observation.
    /// \param observation The value of the observation.
    void new_observation(uint32_t observer_index, double_t observation);

    // ACCESS
    /// \brief Gets the number of variables in the state vector.
    /// \returns The number of variables.
    uint32_t n_variables() const;
    /// \brief Gets the number of observers.
    /// \returns The number of observers.
    uint32_t n_observers() const;
    /// \brief Gets the current state vector.
    /// \returns A const reference to the current state vector.
    const Eigen::VectorXd& state() const;
    /// \brief Gets the current state covariance matrix.
    /// \returns A const reference to the current state covariance matrix.
    const Eigen::MatrixXd& covariance() const;

    // COVARIANCES
    /// \brief The process noise covariance matrix.
    Eigen::MatrixXd Q;
    /// \brief The observation noise covariance matrix.
    Eigen::MatrixXd R;

private:
    // VARIABLES
    /// \brief Stores the actual observations made between iterations.
    std::map<uint32_t, double_t> m_observations;
    
    // DIMENSIONS
    /// \brief The number of variables being estimated by the system.
    uint32_t n_x;
    /// \brief The number of observers.
    uint32_t n_z;

    // STORAGE: PREDICTION
    /// \brief The variable vector.
    Eigen::VectorXd x;
    /// \brief The variable covariance matrix.
    Eigen::MatrixXd P;

    /// \brief The predicted observation vector.
    Eigen::VectorXd z;
    /// \brief The predicted observation covariance.
    Eigen::MatrixXd S;
    /// \brief The innovation cross covariance.
    Eigen::MatrixXd C;
};

}

#endif