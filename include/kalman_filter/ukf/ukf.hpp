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

    // OBSERVER MANAGEMENT
    void add_observer(observer_id_t id, uint32_t dimensions, function_t observation_function);
    void remove_observer(observer_id_t id);

    void initialize(const Eigen::VectorXd& initial_state, const Eigen::MatrixXd& initial_covariance);    

    void predict();
    void update(observer_id_t observer_id, const Eigen::VectorXd& z);

    const Eigen::VectorXd& state() const;
    const Eigen::MatrixXd& covariance() const;

    // COVARIANCES
    /// \brief The process noise covariance matrix.
    Eigen::MatrixXd Q;
    Eigen::MatrixXd& R(observer_id_t observer_id);

    // PARAMETERS
    double_t alpha;
    double_t kappa;
    double_t beta;

private:
    function_t f;
    
    // DIMENSIONS
    /// \brief The number of variables being estimated by the system.
    uint32_t n_x;
    /// \brief The number of elements in the augmented X state.
    uint32_t n_xa;
    /// \brief The number of X sigma points.
    uint32_t n_X;

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
    
    /// \brief The variable covariance sigma matrix (positive half).
    Eigen::MatrixXd Xp;
    /// \brief The process noise sigma matrix (positive half).
    Eigen::MatrixXd Xq;
    /// \brief The evaluated variable sigma matrix.
    Eigen::MatrixXd X;

    struct observer_t
    {
        function_t h;

        uint32_t n_z;
        uint32_t n_za;
        uint32_t n_Z;

        Eigen::VectorXd wm;
        Eigen::VectorXd wc;

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

        /// \brief An interface to the observation noise vector.
        Eigen::VectorXd i_r;
        /// \brief An interface to the predicted observation vector.
        Eigen::VectorXd i_z;

        Eigen::VectorXd t_z;
        Eigen::MatrixXd t_zz;
        Eigen::MatrixXd t_xz;
    };
    std::unordered_map<observer_id_t, observer_t> observers;

    // STORAGE: INTERFACES
    /// \brief An interface to the prior state vector.
    Eigen::VectorXd i_xp;
    /// \brief An interface to the process noise vector.
    Eigen::VectorXd i_q;
    /// \brief An interface to the current state vector.
    Eigen::VectorXd i_x;

    // STORAGE: TEMPORARIES
    Eigen::VectorXd t_x;
    Eigen::MatrixXd t_xx;

    /// \brief An LLT object for storing results of Cholesky decompositions.
    Eigen::LLT<Eigen::MatrixXd> llt;
};

}}

#endif