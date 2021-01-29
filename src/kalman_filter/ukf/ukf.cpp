#include <kalman_filter/ukf/ukf.hpp>

#include <ros/console.h>

using namespace kalman_filter::ukf;

// CONSTRUCTORS
ukf_t::ukf_t(const std::string& model_plugin_path)
{
    // Load the plugin.
    ukf_t::m_model_plugin = model_plugin_t::load_ukf_model(model_plugin_path);

    // Initialize instance.
    ukf_t::initialize();
}
ukf_t::ukf_t(const std::shared_ptr<ukf::model_plugin_t>& model_plugin)
{
    // Store the plugin.
    ukf_t::m_model_plugin = model_plugin;

    // Initialize instance.
    ukf_t::initialize();
}

// METHODS
void ukf_t::initialize()
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

    // Set default values for parameters.
    ukf_t::p_alpha = 0.001;
    ukf_t::p_kappa = 0.0;

    // Call reset to initialize the rest of the members.
    ukf_t::reset();
}
void ukf_t::reset()
{
    // Allocate/initialize all vectors/matrices.
    ukf_t::v_x.setZero(ukf_t::n_variables);
    ukf_t::m_E.setIdentity(ukf_t::n_variables, ukf_t::n_variables);
    ukf_t::v_xa.setZero(ukf_t::n_augmented);
    ukf_t::m_Ea.setZero(ukf_t::n_augmented, ukf_t::n_augmented);
    ukf_t::v_w.setZero(ukf_t::n_sigma);
    ukf_t::m_w.setZero(ukf_t::n_sigma, ukf_t::n_sigma);
    ukf_t::m_X.setZero(ukf_t::n_augmented, ukf_t::n_sigma);
    ukf_t::m_Z.setZero(ukf_t::n_measurements, ukf_t::n_sigma);
    ukf_t::v_z.setZero(ukf_t::n_measurements);
    ukf_t::m_S.setZero(ukf_t::n_measurements, ukf_t::n_measurements);
    ukf_t::m_C.setZero(ukf_t::n_variables, ukf_t::n_measurements);
    ukf_t::m_K.setZero(ukf_t::n_variables, ukf_t::n_measurements);
    ukf_t::i_xp.setZero(ukf_t::n_variables);
    ukf_t::i_x.setZero(ukf_t::n_variables);
    ukf_t::t_ns.setZero(ukf_t::n_variables, ukf_t::n_sigma);;
    ukf_t::t_z.setZero(ukf_t::n_measurements);

    // Use plugin to initialize state.
    ukf_t::m_model_plugin->initialize_state(ukf_t::v_x);

    // Reset state sequence to zero.
    ukf_t::m_sequence = 0;
}
void ukf_t::update()
{
    // Pre-allocate iterator variables.
    uint32_t i, j;

    // Calculate lambda for this iteration (user can change parameters between iterations)
    double lambda = ukf_t::p_alpha * ukf_t::p_alpha * (static_cast<double>(ukf_t::n_augmented) + ukf_t::p_kappa) - static_cast<double>(ukf_t::n_augmented);

    // Calculate weight vector for mean and covariance averaging.
    ukf_t::v_w(0) = lambda / (static_cast<double>(ukf_t::n_augmented) + lambda);
    ukf_t::v_w.tail(ukf_t::n_sigma - 1).fill(1.0 / (2.0 * (static_cast<double>(ukf_t::n_augmented) + lambda)));
    // Create diagonal matrix form of weight vector.
    ukf_t::m_w.diagonal() = ukf_t::v_w;

    // Set up augmented state and covariance matrix.
    // NOTE: Zero values will remain at zero since this is the only assignment of the augmented containers.
    // Copy prior variables into augmented state.
    ukf_t::v_xa.block(0, 0, ukf_t::n_variables, 1) = ukf_t::v_x;
    // Copy prior P, and current Q/R into augmented covariance matrix.
    ukf_t::m_Ea.block(0, 0, ukf_t::n_variables, ukf_t::n_variables) = ukf_t::m_E;
    ukf_t::m_Ea.block(ukf_t::n_variables, ukf_t::n_variables, ukf_t::n_variables, ukf_t::n_variables) = ukf_t::m_model_plugin->q();
    ukf_t::m_Ea.block(2*ukf_t::n_variables, 2*ukf_t::n_variables, ukf_t::n_measurements, ukf_t::n_measurements) = ukf_t::m_model_plugin->r();

    // Generate sigma points for the current augmented state vector.
    // Calculate sqrt((n+lambda)*Ea).
    ukf_t::m_Ea *= (static_cast<double>(ukf_t::n_variables) + lambda);
    // Use Cholesky decomposition to get the square root.
    // Put square root in place of m_Ea since it's not longer needed.
    ukf_t::m_llt.compute(ukf_t::m_Ea);
    ukf_t::m_Ea = ukf_t::m_llt.matrixL();
    // Set first column of the sigma matrix to the variable/mean vector.
    ukf_t::m_X.col(0) = ukf_t::v_xa;
    // Calculate and set reminaing sigma matrix columns.
    for(j = 0; j < ukf_t::n_augmented; ++j)
    {
        ukf_t::m_X.col(j+1) = ukf_t::v_xa + ukf_t::m_Ea.col(j);
        ukf_t::m_X.col(j+1+ukf_t::n_augmented) = ukf_t::v_xa - ukf_t::m_Ea.col(j);
    }

    // Predict new state and covariance.
    // First, transition sigma points through the model plugin.
    // Iterate over each column in the sigma point matrix.
    for(j = 0; j < ukf_t::n_sigma; ++j)
    {
        // Populate the interface vector for prior state.
        // The interface vector is the variable portion of the sigma column + the process noise portion of the sigma column.
        ukf_t::i_xp = ukf_t::m_X.block(0, j, ukf_t::n_variables, 1) + ukf_t::m_X.block(ukf_t::n_variables, j, ukf_t::n_variables, 1);
        // Reset interface vector for new state as a precaution.
        ukf_t::i_x.setZero();
        // Run model plugin's state transition method.
        ukf_t::m_model_plugin->state_transition(ukf_t::i_xp, ukf_t::i_x);
        // Capture transitioned state vector back into X (X not needed after this and need to calculate variance, so can't inline averaging)
        ukf_t::m_X.block(0, j, ukf_t::n_variables, 1) = ukf_t::i_x;
    }
    // Calculate the predicted state as the mean of the transitioned sigma test points for each variable.
    // Using the weight vector as a weighted average across each row of m_X is just matrix multiplication m_X * w
    // This can be done over the prior state to change it to the predicted state.
    ukf_t::v_x.noalias() = ukf_t::m_X.block(0, 0, ukf_t::n_variables, ukf_t::n_sigma) * ukf_t::v_w;
    // Calculate the predicted covariance from the predicted state.
    // This needs to calculate sum of w*(X-x)(X-x)' for each column in the sigma matrix, which is basically (X-x)*m_w*(X-x)'.
    // This can be done over the prioer covariance to change it to the predicted covariance.
    ukf_t::t_ns.noalias() = ukf_t::m_X.block(0, 0, ukf_t::n_variables, ukf_t::n_sigma) * ukf_t::m_w;
    ukf_t::m_E.noalias() = ukf_t::t_ns * ukf_t::m_X.block(0, 0, ukf_t::n_variables, ukf_t::n_sigma).transpose();
}


// ACCESS
const Eigen::VectorXd& ukf_t::state_vector() const
{
    return ukf_t::v_x;
}
uint64_t ukf_t::state_sequence() const
{
    return ukf_t::m_sequence;
}