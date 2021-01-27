#include <kalman_filter/model_plugin.hpp>

#include <ros/console.h>

#include <dlfcn.h>

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
std::shared_ptr<model_plugin_t> model_plugin_t::load(const std::string& path)
{
    // Check that path was provided (dl gets handle to program if empty)
    if(path.empty())
    {
        ROS_ERROR("attempted to load model plugin with empty path");
        return nullptr;
    }

    // Open plugin shared object library.
    void* so_handle = dlopen(path.c_str(), RTLD_NOW);
    if(!so_handle)
    {
        ROS_ERROR_STREAM("failed to load model plugin (" << dlerror() << ")");
        return nullptr;
    }

    // Get a reference to the instantiate symbol.
    typedef model_plugin_t* (*instantiate_t)();
    instantiate_t instantiate = reinterpret_cast<instantiate_t>(dlsym(so_handle, "instantiate"));
    if(!instantiate)
    {
        ROS_ERROR_STREAM("failed to load model plugin (" << dlerror() << ")");
        dlclose(so_handle);
        return nullptr;
    }

    // Try to instantiate the plugin.
    model_plugin_t* plugin = nullptr;
    try
    {
        plugin = instantiate();
    }
    catch(const std::exception& error)
    {
        ROS_ERROR_STREAM("failed to instantiate model plugin (" << error.what() << ")");
        dlclose(so_handle);
        return nullptr;
    }

    // Return the plugin as a shared ptr with a custom deleter.
    return std::shared_ptr<model_plugin_t>(plugin,
                                           [so_handle](model_plugin_t* plugin){delete plugin; dlclose(so_handle);});
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

// METHODS
void model_plugin_t::initialize_state(Eigen::VectorXd& state)
{
    // For base class, set initial state to zero.
    state.setZero();
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