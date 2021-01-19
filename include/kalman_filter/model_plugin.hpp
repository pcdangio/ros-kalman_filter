#ifndef KALMAN_FILTER___MODEL_PLUGIN_H
#define KALMAN_FILTER___MODEL_PLUGIN_H

#include <eigen3/Eigen/Dense>

#include <mutex>

namespace kalman_filter {

class model_plugin_t
{
public:
    model_plugin_t(uint32_t n_state_variables, uint32_t n_measurement_variables);

    uint32_t n_state_variables() const;
    uint32_t n_measurement_variables() const;

    void get_process_covariance(Eigen::MatrixXd& q) const;
    void get_measurement_covariance(Eigen::MatrixXd& r) const;
    void get_measurements(Eigen::VectorXd& z, Eigen::VectorXd& availability) const;
    
    virtual void state_transition(const Eigen::VectorXd& current_state, Eigen::VectorXd& next_state) const = 0;
    virtual void predict_measurement(const Eigen::VectorXd& state, Eigen::VectorXd& measurement) const = 0;

protected:
    // PROCESS COVARIANCE MATRIX
    Eigen::MatrixXd m_process_covariance;
    void lock_process_covariance() const;
    void unlock_process_covariance() const;

    // MEASUREMENT COVARIANCE MATRIX
    Eigen::MatrixXd m_measurement_covariance;
    void lock_measurement_covariance() const;
    void unlock_measurement_covariance() const;

    // MEASUREMENTS
    void new_measurement(uint32_t index, double value);
    void lock_measurement() const;
    void unlock_measurement() const;
    
private:
    uint32_t m_n_state_variables;
    uint32_t m_n_measurement_variables;
    
    mutable std::mutex m_mutex_process_covariance;
    mutable std::mutex m_mutex_measurement_covariance;
    mutable std::mutex m_mutex_measurement;

    Eigen::VectorXd m_measurement;
    Eigen::VectorXd m_measurement_availability;
};

}

#endif