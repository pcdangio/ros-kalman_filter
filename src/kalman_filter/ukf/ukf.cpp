#include <kalman_filter/ukf/ukf.hpp>

#include <ros/console.h>

using namespace kalman_filter::ukf;

ukf_t::ukf_t(const std::string& model_plugin_path)
{
    // Load the plugin.
    ukf_t::m_model_plugin = model_plugin_t::load_ukf_model(model_plugin_path);

    // Use the reset method to initialize.
    ukf_t::reset();
}
ukf_t::ukf_t(const std::shared_ptr<ukf::model_plugin_t>& model_plugin)
{
    // Store the plugin.
    ukf_t::m_model_plugin = model_plugin;

    // Use the reset method to initialize.
    ukf_t::reset();
}

void ukf_t::update()
{

}
void ukf_t::reset()
{
    // Check if model plugin is valid.
    if(ukf_t::m_model_plugin)
    {
        // Retrieve and store variable sizes.
        ukf_t::n_variables = ukf_t::m_model_plugin->n_state_variables();
        ukf_t::n_measurements = ukf_t::m_model_plugin->n_measurement_variables();
    }
    else
    {
        // Set variable sizes to zero.
        ukf_t::n_variables = 0;
        ukf_t::n_measurements = 0;
        // Log error.
        ROS_ERROR("ukf has not been supplied with a valid model plugin");
    }
    
    // Calculate augmented state size and number of sigma points.
    ukf_t::n_augmented = ukf_t::n_variables*2 + ukf_t::n_measurements;
    ukf_t::n_sigma = 2*ukf_t::n_augmented + 1;

    // Allocate/initialize all vectors/matrices.
    ukf_t::v_x.setZero(ukf_t::n_variables);
    ukf_t::m_E.setIdentity(ukf_t::n_variables, ukf_t::n_variables);
    ukf_t::v_xa.setZero(ukf_t::n_augmented);
    ukf_t::m_Ea.setZero(ukf_t::n_augmented, ukf_t::n_augmented);
    ukf_t::m_X.setZero(ukf_t::n_augmented, ukf_t::n_sigma);
    ukf_t::m_Z.setZero(ukf_t::n_measurements, ukf_t::n_sigma);
    ukf_t::v_z.setZero(ukf_t::n_measurements);
    ukf_t::m_S.setZero(ukf_t::n_measurements, ukf_t::n_measurements);
    ukf_t::m_C.setZero(ukf_t::n_variables, ukf_t::n_measurements);
    ukf_t::m_K.setZero(ukf_t::n_variables, ukf_t::n_measurements);
    ukf_t::t_x.setZero(ukf_t::n_variables);
    ukf_t::t_z.setZero(ukf_t::n_measurements);

    // Use plugin to initialize state.
    ukf_t::m_model_plugin->initialize_state(ukf_t::v_x);

    // Reset state sequence to zero.
    ukf_t::m_sequence = 0;
}

const Eigen::VectorXd& ukf_t::state_vector() const
{
    return ukf_t::v_x;
}
uint64_t ukf_t::state_sequence() const
{
    return ukf_t::m_sequence;
}