#ifndef KALMAN_FILTER___UKF___UKF_H
#define KALMAN_FILTER___UKF___UKF_H

#include <eigen3/Eigen/Dense>

#include <functional>
#include <unordered_map>

namespace kalman_filter {
namespace ukf {

typedef uint32_t observer_id_t;
typedef std::function<void(const Eigen::VectorXd&, const Eigen::VectorXd&, Eigen::VectorXd&)> function_t;

class ukf_t
{
public:
    // CONSTRUCTORS
    ukf_t(uint32_t dimensions, function_t prediction_function);

    bool add_observer(observer_id_t id, uint32_t dimensions, function_t observation_function);
    bool remove_observer(observer_id_t id);

    void initialize(const Eigen::VectorXd& initial_state, const Eigen::MatrixXd& initial_covariance);    

    void predict();
    void update(observer_id_t id, const Eigen::VectorXd& z);

    const Eigen::VectorXd& state() const;

    // COVARIANCES
    /// \brief The process noise covariance matrix.
    Eigen::MatrixXd Q;
    Eigen::MatrixXd* R(observer_id_t id);

    // PARAMETERS
    double_t alpha;
    double_t kappa;
    double_t beta;

private:
    function_t f;
    
    // DIMENSIONS
    /// \brief The number of variables being estimated by the system.
    uint32_t n_variables;
    /// \brief The number of X sigma points.
    uint32_t n_sigma_x;
    /// \brief The number of Q sigma points.
    uint32_t n_sigma_q;

    // WEIGHTS
    /// \brief The mean recovery weight vector.
    Eigen::VectorXd wm;
    /// \brief The covariance recovery weight vector.
    Eigen::VectorXd wc;

    // STORAGE: VARIABLE / PROCESS NOISE
    /// \brief The variable vector.
    Eigen::VectorXd x;
    /// \brief The variable covariance matrix.
    Eigen::MatrixXd P;
    
    /// \brief The variable sigma matrix.
    Eigen::MatrixXd Xx;
    /// \brief The process noise sigma matrix.
    Eigen::MatrixXd Xq;
    /// \brief The evaluated variable sigma matrix.
    Eigen::MatrixXd X;

    struct observer_t
    {
        function_t h;

        uint32_t n_observers;
        uint32_t n_sigma_z;

        Eigen::VectorXd wm;
        Eigen::VectorXd wc;

        /// \brief The observation noise covariance matrix.
        Eigen::MatrixXd R;
        /// \brief The observation noise sigma matrix.
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

        /// \brief An interface to the observation noise vector.
        Eigen::VectorXd i_r;
        /// \brief An interface to the predicted observation vector.
        Eigen::VectorXd i_z;
    };
    std::unordered_map<observer_id_t, observer_t> observers;

    // STORAGE: INTERFACES
    /// \brief An interface to the prior state vector.
    Eigen::VectorXd i_xp;
    /// \brief An interface to the process noise vector.
    Eigen::VectorXd i_q;
    /// \brief An interface to the current state vector.
    Eigen::VectorXd i_x;

    /// \brief An LLT object for storing results of Cholesky decompositions.
    Eigen::LLT<Eigen::MatrixXd> llt;
};

}}

#endif