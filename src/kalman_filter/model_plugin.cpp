#include <kalman_filter/model_plugin.hpp>

using namespace kalman_filter;

// CONSTRUCTORS
model_plugin_t::model_plugin_t(uint32_t n_state_variables, uint32_t n_measurement_variables)
{
    // Store state/measurement vector sizes.
    model_plugin_t::m_n_state_variables = n_state_variables;
    model_plugin_t::m_n_measurement_variables = n_measurement_variables;

    // Initialize process and measurement covariance matrices.
    model_plugin_t::m_process_covariance.setIdentity(n_state_variables, n_state_variables);
    model_plugin_t::m_measurement_covariance.setIdentity(n_measurement_variables, n_measurement_variables);

    // Initialize measurement vectors.
    model_plugin_t::m_measurement.setZero(n_measurement_variables);
    model_plugin_t::m_measurement_availability.setZero(n_measurement_variables);
}

// PROPERTIES
uint32_t model_plugin_t::n_state_variables() const
{
    return model_plugin_t::m_n_state_variables;
}
uint32_t model_plugin_t::n_measurement_variables() const
{
    return model_plugin_t::m_n_measurement_variables;
}


// GET
void model_plugin_t::get_process_covariance(Eigen::MatrixXd& q) const
{
    model_plugin_t::m_mutex_process_covariance.lock();
    q = model_plugin_t::m_process_covariance;
    model_plugin_t::m_mutex_process_covariance.unlock();
}
void model_plugin_t::get_measurement_covariance(Eigen::MatrixXd& r) const
{
    model_plugin_t::m_mutex_measurement_covariance.lock();
    r = model_plugin_t::m_measurement_covariance;
    model_plugin_t::m_mutex_measurement_covariance.unlock();
}
void model_plugin_t::get_measurements(Eigen::VectorXd& z, Eigen::VectorXd& availability) const
{
    model_plugin_t::m_mutex_measurement.lock();
    z = model_plugin_t::m_measurement;
    availability = model_plugin_t::m_measurement_availability;
    model_plugin_t::m_mutex_measurement.unlock();
}

// MEASUREMENTS
void model_plugin_t::new_measurement(uint32_t index, double value)
{

}

// LOCKING
void model_plugin_t::lock_process_covariance() const
{
    model_plugin_t::m_mutex_process_covariance.lock();
}
void model_plugin_t::unlock_process_covariance() const
{
    model_plugin_t::m_mutex_process_covariance.unlock();
}
void model_plugin_t::lock_measurement_covariance() const
{
    model_plugin_t::m_mutex_measurement_covariance.lock();
}
void model_plugin_t::unlock_measurement_covariance() const
{
    model_plugin_t::m_mutex_measurement_covariance.unlock();
}
void model_plugin_t::lock_measurement() const
{
    model_plugin_t::m_mutex_measurement.lock();
}
void model_plugin_t::unlock_measurement() const
{
    model_plugin_t::m_mutex_measurement.unlock();
}