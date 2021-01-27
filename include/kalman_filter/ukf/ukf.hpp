#ifndef KALMAN_FILTER___UKF___UKF_H
#define KALMAN_FILTER___UKF___UKF_H

#include <kalman_filter/ukf/model_plugin.hpp>

namespace kalman_filter {
namespace ukf {

class ukf_t
{
public:
    ukf_t(const std::string& model_plugin_path);
    ~ukf_t();

    void iterate();
    void reset();

    const Eigen::VectorXd& state_vector() const;
    uint64_t state_sequence() const;

private:
    ukf::model_plugin_t* m_model;
    void* m_plugin_handle;

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
    /// \brief A temporary vector of size n_variables.
    Eigen::VectorXd t_x;
    /// \brief A temporary vector of size n_measurements.
    Eigen::VectorXd t_z;

    // SEQUENCE TRACKING
    uint64_t m_sequence;
};

}}

#endif