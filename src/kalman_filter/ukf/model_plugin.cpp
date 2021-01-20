#include <kalman_filter/ukf/model_plugin.hpp>

using namespace kalman_filter::ukf;

model_plugin_t::model_plugin_t(uint32_t n_state_variables, uint32_t n_measurement_variables)
    : kalman_filter::model_plugin_t(n_state_variables, n_measurement_variables)
{

}