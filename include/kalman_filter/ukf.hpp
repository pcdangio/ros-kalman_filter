/// \file kalman_filter/ukf.hpp
/// \brief Defines the kalman_filter::ukf_t class.
#ifndef KALMAN_FILTER___UKF_H
#define KALMAN_FILTER___UKF_H

#include <eigen3/Eigen/Dense>

#include <functional>
#include <unordered_map>

/// \brief Includes objects for Kalman Filtering.
namespace kalman_filter {

/// \brief An Unscented Kalman Filter.
class ukf_t
{
public:
    // CONSTRUCTORS
    /// \brief Instantiates a new ukf_t object.
    /// \param n_variables The number of variables in the state vector.
    /// \param n_observers The number of state observers.
    ukf_t(uint32_t n_variables, uint32_t n_observers);

    // MODEL FUNCTIONS
    /// \brief Predicts a new state by transitioning from a prior state.
    /// \param xp The prior state to transition from.
    /// \param q The prediction's noise vector.
    /// \param x (OUTPUT) The predicted new state.
    /// \note This function must not make changes to any external object.
    virtual void state_transition(const Eigen::VectorXd& xp, const Eigen::VectorXd& q, Eigen::VectorXd& x) const = 0;
    /// \brief Predicts an observation from a state.
    /// \param x The state to predict an observation from.
    /// \param r The prediction's noise vector.
    /// \param z (OUTPUT) The predicted observation.
    /// \note This function must not make changes to any external object.
    virtual void observation(const Eigen::VectorXd& x, const Eigen::VectorXd& r, Eigen::VectorXd& z) const = 0;

    // FILTER METHODS
    /// \brief Initializes the UKF with a specified state and covariance.
    /// \param initial_state The initial state vector to initialize with.
    /// \param initial_covariance The initial state covariance to initialize with.
    void initialize_state(const Eigen::VectorXd& initial_state, const Eigen::MatrixXd& initial_covariance);
    /// \brief Predicts a new state and performs update corrections with available observations.
    /// \note The iteration rate should be at least as fast as the fastest observer rate.
    void iterate();
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

    // PARAMETERS
    /// \brief The alpha parameter of the UKF.
    double_t alpha;
    /// \brief The kappa parameter of the UKF.
    double_t kappa;
    /// \brief The beta parameter of the UKF.
    double_t beta;

private:
    // VARIABLES
    /// \brief Stores the actual observations made between iterations.
    std::unordered_map<uint32_t, double_t> m_observations;
    
    // DIMENSIONS
    /// \brief The number of variables being estimated by the system.
    uint32_t n_x;
    /// \brief The number of observers.
    uint32_t n_z;
    /// \brief The number of variables in the augemented state (x q z).
    uint32_t n_a;
    /// \brief The number of sigma points.
    uint32_t n_s;

    // STORAGE: WEIGHTS
    /// \brief The mean recovery weight vector.
    Eigen::VectorXd wm;
    /// \brief The covariance recovery weight vector.
    Eigen::VectorXd wc;

    // STORAGE: PREDICTION
    /// \brief The variable vector.
    Eigen::VectorXd x;
    /// \brief The variable covariance matrix.
    Eigen::MatrixXd P;
    /// \brief The variable covariance sigma matrix (positive half).
    Eigen::MatrixXd Xp;
    /// \brief The process noise sigma matrix (positive half).
    Eigen::MatrixXd Xq;
    /// \brief The evaluated variable sigma matrix.
    Eigen::MatrixXd X;

    // STORAGE: UPDATE
    /// \brief The observation noise sigma matrix (positive half).
    Eigen::MatrixXd Xr;
    /// \brief The evaluated observation sigma matrix.
    Eigen::MatrixXd Z;
    /// \brief The predicted observation vector.
    Eigen::VectorXd z;
    /// \brief The predicted observation covariance.
    Eigen::MatrixXd S;
    /// \brief The innovation cross covariance.
    Eigen::MatrixXd C;

    // STORAGE: INTERFACES
    /// \brief An interface to the prior state vector.
    Eigen::VectorXd i_xp;
    /// \brief An interface to the process noise vector.
    Eigen::VectorXd i_q;
    /// \brief An interface to the current state vector.
    Eigen::VectorXd i_x;
    /// \brief An interface to the observation noise vector.
    Eigen::VectorXd i_r;
    /// \brief An interface to the predicted observation vector.
    Eigen::VectorXd i_z;

    // STORAGE: TEMPORARIES
    /// \brief A temporary working vector of size x.
    Eigen::VectorXd t_x;
    /// \brief A temporary working matrix of size x,x.
    Eigen::MatrixXd t_xx;
    /// \brief A temporary working vector of size z.
    Eigen::VectorXd t_z;
    /// \brief A temporary working matrix of size z,z.
    Eigen::MatrixXd t_zz;
    /// \brief A temporary working matrix of size x,z.
    Eigen::MatrixXd t_xz;

    // UTILITY
    /// \brief An LLT object for storing results of Cholesky decompositions.
    mutable Eigen::LLT<Eigen::MatrixXd> llt;

    // UTILITY FUNCTIONS
    /// \brief Calculates scaling parameters.
    /// \param n_a The number of augmented states.
    /// \param lambda (OUTPUT) The resulting lambda.
    /// \param wm (OUTPUT) The resulting mean recovery weight vector.
    /// \param wc (OUTPUT) The resulting covariance recovery weight vector.
    void calculate_scaling(uint32_t n_a, double_t& lambda, Eigen::VectorXd& wm, Eigen::VectorXd& wc) const;
    /// \brief Calculates a particular sigma matrix component.
    /// \param n_a The number of augmented states.
    /// \param lambda The lambda to use for the calculation.
    /// \param covariance The covariance matrix to derive the sigma component from.
    /// \param sigma_component (OUTPUT) The resulting sigma matrix component.
    void populate_sigma_component(uint32_t n_a, double_t lambda, const Eigen::MatrixXd& covariance, Eigen::MatrixXd& sigma_component) const;
};

}

#endif