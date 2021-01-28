#include <kalman_filter/ukf/model_plugin.hpp>

using namespace kalman_filter::ukf;

model_plugin_t::model_plugin_t(uint32_t n_state_variables, uint32_t n_measurement_variables)
    : kalman_filter::model_plugin_t(n_state_variables, n_measurement_variables)
{

}
std::shared_ptr<kalman_filter::ukf::model_plugin_t> model_plugin_t::load_ukf_model(const std::string& plugin_path)
{
    return std::dynamic_pointer_cast<kalman_filter::ukf::model_plugin_t>(model_plugin_t::load_base_model(plugin_path));
}