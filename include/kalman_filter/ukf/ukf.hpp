#ifndef KALMAN_FILTER___UKF___UKF_H
#define KALMAN_FILTER___UKF___UKF_H

#include <kalman_filter/ukf/model_plugin.hpp>

namespace kalman_filter {
namespace ukf {

class ukf_t
{
public:
    ukf_t(const std::string& model_plugin_path);
    ukf_t(const std::shared_ptr<ukf::model_plugin_t>& model_plugin);

    void update();
    void reset();

    const Eigen::VectorXd& state_vector() const;
    uint64_t state_sequence() const;

    // PARAMETERS
    double p_alpha;
    double p_kappa;

private:
    std::shared_ptr<ukf::model_plugin_t> m_model_plugin;

    // DIMENSIONS
    /// \brief The number of variables being estimated by the system.
    uint32_t n_variables;
    /// \brief The number of variable measurements.
    uint32_t n_measurements;
    /// \brief The total size of the UKF's augmented state vector.
    uint32_t n_augmented;
    /// \brief The total number of sigma points used by the UKF.
    uint32_t n_sigma;

    // RUNTIME MATRICES
    /// \brief The variable vector.
    Eigen::VectorXd v_x;
    /// \brief The variable covariance matrix.
    Eigen::MatrixXd m_E;
    /// \brief The augmented state vector.
    Eigen::VectorXd v_xa;
    /// \brief The augmented covariance matrix.
    Eigen::MatrixXd m_Ea;
    /// \brief The weight vector for weighted averaging.
    Eigen::VectorXd v_w;
    /// \brief The weight matrix for weighted averaging.
    Eigen::MatrixXd m_w;
    /// \brief The sigma point variable matrix.
    Eigen::MatrixXd m_X;
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

    // SEQUENCE TRACKING
    uint64_t m_sequence;

    void initialize();
};

}}

#endif