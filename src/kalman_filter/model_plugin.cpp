#include <kalman_filter/model_plugin.hpp>

using namespace kalman_filter;

// CONSTRUCTORS
model_plugin_t::model_plugin_t(uint32_t n_state_variables, uint32_t n_measurement_variables)
{
    // Store state/measurement vector sizes.
    model_plugin_t::m_n_state_variables = n_state_variables;
    model_plugin_t::m_n_measurement_variables = n_measurement_variables;

    // Initialize process and measurement covariance matrices.
    model_plugin_t::m_q.setIdentity(n_state_variables, n_state_variables);
    model_plugin_t::m_r.setIdentity(n_measurement_variables, n_measurement_variables);

    // Initialize measurement vector and mask.
    model_plugin_t::m_z.setZero(n_measurement_variables);
    model_plugin_t::m_m.setZero(n_measurement_variables, n_measurement_variables);
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

// ACCESS
const Eigen::MatrixXd& model_plugin_t::q() const
{
    return model_plugin_t::m_q;
}
const Eigen::MatrixXd& model_plugin_t::r() const
{
    return model_plugin_t::m_r;
}
const Eigen::VectorXd& model_plugin_t::z() const
{
    return model_plugin_t::m_z;
}
const Eigen::MatrixXd& model_plugin_t::m() const
{
    return model_plugin_t::m_m;
}

// MEASUREMENT
void model_plugin_t::new_measurement(uint32_t index, double value)
{
    // Store value in measurement vector.
    model_plugin_t::m_z(index) = value;
    // Update measurement mask matrix.
    model_plugin_t::m_m(index, index) = 1.0;
}
void model_plugin_t::clear_measurements()
{
    // Reset measurement vector and mask to zero.
    model_plugin_t::m_z.setZero();
    model_plugin_t::m_m.setZero();
}