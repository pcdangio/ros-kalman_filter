#ifndef KALMAN_FILTER___UKF___UKF_H
#define KALMAN_FILTER___UKF___UKF_H

#include <kalman_filter/ukf/model.hpp>

namespace kalman_filter {
namespace ukf {

class ukf_t
{
public:
    ukf_t(const std::shared_ptr<model_t>& model);

    void initialize(const Eigen::VectorXd& initial_state);

    void predict();
    void update(observer_id_t observer, const Eigen::VectorXd& z);

    const Eigen::VectorXd& state_vector() const;

    // PARAMETERS
    double p_alpha;
    double p_kappa;

private:
    std::shared_ptr<ukf::model_t> m_model;

    // DIMENSIONS
    /// \brief The number of variables being estimated by the system.
    uint32_t n_variables;
    uint32_t n_sigma_x;
    uint32_t n_sigma_q;

    // // MODEL COMPONENTS
    // /// \brief A reference to the model's process noise matrix.
    // Eigen::MatrixXd& m_Q;
    // /// \brief A reference to the model's measurement noise matrix.
    // Eigen::MatrixXd& m_R;
    // std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> f_state_transition;
    // std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> f_measurement_update;

    // STORAGE: VARIABLE / PROCESS NOISE
    /// \brief The variable vector.
    Eigen::VectorXd x;
    /// \brief The variable covariance matrix.
    Eigen::MatrixXd P;
    /// \brief The mean recovery weight vector.
    Eigen::VectorXd wm;
    /// \brief The covariance recovery weight vector.
    Eigen::VectorXd wc;
    /// \brief The variable sigma matrix.
    Eigen::MatrixXd Xx;
    /// \brief The process noise sigma matrix.
    Eigen::MatrixXd Xq;
    /// \brief The evaluated variable sigma matrix.
    Eigen::MatrixXd X;

    /// \brief The sigma point matrix for the predicted measurement.
    Eigen::MatrixXd m_Z;
    /// \brief The predicted measurement mean.
    Eigen::VectorXd v_z;
    /// \brief The predicted measurement covariance.
    Eigen::MatrixXd m_S;
    /// \brief The predicted cross covariance.
    Eigen::MatrixXd m_C;
    /// \brief The Kalman gain matrix.
    Eigen::MatrixXd m_K;
    /// \brief An plugin interface vector for the prior state.
    Eigen::VectorXd i_xp;
    /// \brief A plugin interface vector for the predicted state.
    Eigen::VectorXd i_x;
    /// \brief A temporary vector of size n_variables * n_sigma.
    Eigen::VectorXd t_ns;
    /// \brief A temporary vector of size n_measurements.
    Eigen::VectorXd t_z;

    /// \brief An LLT object for storing results of Cholesky decompositions.
    Eigen::LLT<Eigen::MatrixXd> m_llt;


    void initialize();
};

}}

#endif