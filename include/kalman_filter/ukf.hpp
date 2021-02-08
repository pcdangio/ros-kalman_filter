/// \file kalman_filter/ukf.hpp
/// \brief Defines the kalman_filter::ukf_t class.
#ifndef KALMAN_FILTER___UKF_H
#define KALMAN_FILTER___UKF_H

#include <eigen3/Eigen/Dense>

#include <functional>
#include <unordered_map>

/// \brief Includes objects for Kalman Filtering.
namespace kalman_filter {

// TYPE DEFINITIONS
/// \brief A unique ID number for an observer.
typedef uint32_t observer_id_t;
/// \brief A callback signature for state transition and observation functions.
typedef std::function<void(const Eigen::VectorXd&, const Eigen::VectorXd&, Eigen::VectorXd&)> function_t;

/// \brief An Unscented Kalman Filter.
class ukf_t
{
public:
    // CONSTRUCTORS
    /// \brief Instantiates a new ukf_t object.
    /// \param dimensions The number of dimensions in the state vector.
    /// \param prediction_function The function for predicting a new state from the current state.
    ukf_t(uint32_t dimensions, function_t prediction_function);

    // OBSERVER MANAGEMENT
    /// \brief Adds a new observer to the UKF.
    /// \param id The unique ID to assign the observer.
    /// \param dimensions The number of dimensions in the observation vector.
    /// \param observation_function The function for calculating the observation vector from the current state.
    /// \note If an existing ID is provided, the existing observer will be replaced.
    void add_observer(observer_id_t id, uint32_t dimensions, function_t observation_function);
    /// \brief Removes an observer from the UKF.
    /// \param id The unique ID of the observer to remove.
    void remove_observer(observer_id_t id);
    /// \brief Removes all observers from the UKF.
    void clear_observers();

    // FILTER METHODS
    /// \brief Initializes the UKF with a specified state and covariance.
    /// \param initial_state The initial state vector to initialize with.
    /// \param initial_covariance The initial state covariance to initialize with.
    void initialize_state(const Eigen::VectorXd& initial_state, const Eigen::MatrixXd& initial_covariance);    
    /// \brief Predicts a new state from the current state.
    void predict();
    /// \brief Updates the current state with an observation.
    /// \param observer_id The unique ID of the observer providing the observation.
    /// \param z The vector of observations from the observer.
    void update(observer_id_t observer_id, const Eigen::VectorXd& z);

    // ACCESS
    /// \brief Gets the number of variables in the state vector.
    /// \returns The number of variables.
    uint32_t n_variables() const;
    /// \brief Gets the current state vector.
    /// \returns A const reference to the current state vector.
    const Eigen::VectorXd& state() const;
    /// \brief Gets the current state covariance matrix.
    /// \returns A const reference to the current state covariance matrix.
    const Eigen::MatrixXd& covariance() const;

    // COVARIANCES
    /// \brief The process noise covariance matrix.
    Eigen::MatrixXd Q;
    /// \brief The observation noise covariance matrix for an observer.
    /// \param observer_id The unique ID of the observer to access R from.
    /// \returns A reference to the observer's noise covariance matrix.
    Eigen::MatrixXd& R(observer_id_t observer_id);

    // PARAMETERS
    /// \brief The alpha parameter of the UKF.
    double_t alpha;
    /// \brief The kappa parameter of the UKF.
    double_t kappa;
    /// \brief The beta parameter of the UKF.
    double_t beta;

private:
    // FUNCTIONS
    /// \brief The state transition function.
    function_t f;
    
    // DIMENSIONS
    /// \brief The number of variables being estimated by the system.
    uint32_t n_x;
    /// \brief The number of elements in the augmented X state.
    uint32_t n_xa;
    /// \brief The number of X sigma points.
    uint32_t n_X;

    // STORAGE: WEIGHTS
    /// \brief The mean recovery weight vector.
    Eigen::VectorXd wm;
    /// \brief The covariance recovery weight vector.
    Eigen::VectorXd wc;

    // STORAGE: VARIABLE / PROCESS NOISE
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

    // STORAGE: INTERFACES
    /// \brief An interface to the prior state vector.
    Eigen::VectorXd i_xp;
    /// \brief An interface to the process noise vector.
    Eigen::VectorXd i_q;
    /// \brief An interface to the current state vector.
    Eigen::VectorXd i_x;

    // STORAGE: TEMPORARIES
    /// \brief A temporary working vector of size x.
    Eigen::VectorXd t_x;
    /// \brief A temporary working matrix of size x,x.
    Eigen::MatrixXd t_xx;

    // UTILITY
    /// \brief An LLT object for storing results of Cholesky decompositions.
    Eigen::LLT<Eigen::MatrixXd> llt;

    // OBSERVER
    /// \brief An observer for making observations.
    struct observer_t
    {
        // FUNCTIONS
        /// \brief The observation/H function.
        function_t h;

        // DIMENSIONS
        /// \brief The number of observations provided by the observer.
        uint32_t n_z;
        /// \brief The number of elements in the augmented X state.
        uint32_t n_za;
        /// \brief The number of Z sigma points.
        uint32_t n_Z;

        // STORAGE: WEIGHTS
        /// \brief The mean recovery weight vector.
        Eigen::VectorXd wm;
        /// \brief The covariance recovery weight vector.
        Eigen::VectorXd wc;

        // STORAGE: OBSERVATIONS & UPDATE
        /// \brief The observation noise covariance matrix.
        Eigen::MatrixXd R;
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
        /// \brief The Kalman gain.
        Eigen::MatrixXd K;

        // STORAGE: INTERFACES
        /// \brief An interface to the observation noise vector.
        Eigen::VectorXd i_r;
        /// \brief An interface to the predicted observation vector.
        Eigen::VectorXd i_z;

        // STORAGE: TEMPORARIES
        /// \brief A temporary working vector of size z.
        Eigen::VectorXd t_z;
        /// \brief A temporary working matrix of size z,z.
        Eigen::MatrixXd t_zz;
        /// \brief A temporary working matrix of size x,z.
        Eigen::MatrixXd t_xz;
    };
    /// \brief The UKF's observer instances.
    std::unordered_map<observer_id_t, observer_t> observers;
};

}

#endif