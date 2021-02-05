#include <kalman_filter/ukf/ukf.hpp>

using namespace kalman_filter::ukf;

// CONSTRUCTORS
ukf_t::ukf_t(uint32_t dimensions, function_t prediction_function)
{
    // Store prediction function.
    ukf_t::f = prediction_function;

    // Set default parameters.
    ukf_t::alpha = 0.001;
    ukf_t::kappa = 3 - n_variables;
    ukf_t::beta = 2;

    // Store dimension sizes.
    ukf_t::n_variables = dimensions;
    ukf_t::n_sigma_x = 2*ukf_t::n_variables + 1;
    ukf_t::n_sigma_q = 2*ukf_t::n_variables + 1;

    // Allocate weight vectors.
    ukf_t::wm.setZero(ukf_t::n_sigma_x + ukf_t::n_sigma_q);
    ukf_t::wc.setZero(ukf_t::n_sigma_x + ukf_t::n_sigma_q);

    // Allocate variable components.
    ukf_t::x.setZero(ukf_t::n_variables);
    ukf_t::P.setIdentity(ukf_t::n_variables, ukf_t::n_variables);
    ukf_t::Q.setIdentity(ukf_t::n_variables, ukf_t::n_variables);
    ukf_t::Xx.setZero(ukf_t::n_variables, ukf_t::n_sigma_x);
    ukf_t::Xq.setZero(ukf_t::n_variables, ukf_t::n_sigma_q);
    ukf_t::X.setZero(ukf_t::n_variables, ukf_t::n_sigma_x + ukf_t::n_sigma_q);

    // Allocate interface components.
    ukf_t::i_xp.setZero(ukf_t::n_variables);
    ukf_t::i_x.setZero(ukf_t::n_variables);
    ukf_t::i_q.setZero(ukf_t::n_variables);

    // Log instantiation.
    ROS_INFO_STREAM("instantiated ukf with [" << dimensions << "] dimensional state vector");
}

// INITIALIZATION
void ukf_t::add_observer(observer_id_t id, uint32_t dimensions, function_t observation_function)
{
    // Create a new observer and grab a reference to it (or grab existing observer to replace it)
    observer_t& observer = ukf_t::observers[id];

    // Store observer's function.
    observer.h = observation_function;

    // Store dimension sizes.
    observer.n_observers = dimensions;
    observer.n_sigma_z = 2*observer.n_observers + 1;

    // Allocate weight vectors.
    observer.wm.setZero(ukf_t::n_sigma_x + observer.n_sigma_z);
    observer.wc.setZero(ukf_t::n_sigma_x + observer.n_sigma_z);

    // Allocate observer components.
    observer.R.setIdentity(observer.n_observers, observer.n_observers);
    observer.Xr.setZero(observer.n_observers, observer.n_sigma_z);
    observer.Z.setZero(observer.n_observers, ukf_t::n_sigma_x + observer.n_sigma_z);
    observer.z.setZero(observer.n_observers);
    observer.S.setZero(observer.n_observers, observer.n_observers);
    observer.C.setZero(ukf_t::n_variables, observer.n_observers);
    observer.K.setZero(ukf_t::n_variables, observer.n_observers);

    // Allocate interface components.
    observer.i_r.setZero(observer.n_observers);
    observer.i_z.setZero(observer.n_observers);
}
void ukf_t::remove_observer(observer_id_t id)
{
    // Remove observer.
    ukf_t::observers.erase(id)
}
void ukf_t::initialize(const Eigen::VectorXd& initial_state, const Eigen::MatrixXd& initial_covariance)
{
    // Verify initial state size.
    if(initial_state.size() != ukf_t::n_variables)
    {
        throw std::runtime_error("failed to initialize state vector (initial state provided has incorrect size)");
    }

    // Verify initial covariance size.
    if(initial_covariance.rows() != ukf_t::n_variables || initial_covariance.cols() != ukf_t::n_variables)
    {
        throw std::runtime_error("failed to initialize state covariance (initial covariance provided has incorrect size)");
    }

    // Copy initial state and covariance.
    ukf_t::x = initial_state;
    ukf_t::P = initial_covariance;
}

void ukf_t::predict()
{

}
void ukf_t::update(observer_id_t observer, const Eigen::VectorXd& z)
{

}

const Eigen::VectorXd& ukf_t::state() const
{
    return ukf_t::x;
}





// void ukf_t::update()
// {
//     // Pre-allocate iterator variables.
//     uint32_t i, j;

//     // Calculate lambda for this iteration (user can change parameters between iterations)
//     double lambda = ukf_t::p_alpha * ukf_t::p_alpha * (static_cast<double>(ukf_t::n_augmented) + ukf_t::p_kappa) - static_cast<double>(ukf_t::n_augmented);

//     // Calculate weight vector for mean and covariance averaging.
//     ukf_t::v_w(0) = lambda / (static_cast<double>(ukf_t::n_augmented) + lambda);
//     ukf_t::v_w.tail(ukf_t::n_sigma - 1).fill(1.0 / (2.0 * (static_cast<double>(ukf_t::n_augmented) + lambda)));
//     // Create diagonal matrix form of weight vector.
//     ukf_t::m_w.diagonal() = ukf_t::v_w;

//     // Set up augmented state and covariance matrix.
//     // NOTE: Zero values will remain at zero since this is the only assignment of the augmented containers.
//     // Copy prior variables into augmented state.
//     ukf_t::v_xa.block(0, 0, ukf_t::n_variables, 1) = ukf_t::v_x;
//     // Copy prior P, and current Q/R into augmented covariance matrix.
//     ukf_t::m_Ea.block(0, 0, ukf_t::n_variables, ukf_t::n_variables) = ukf_t::m_E;
//     ukf_t::m_Ea.block(ukf_t::n_variables, ukf_t::n_variables, ukf_t::n_variables, ukf_t::n_variables) = ukf_t::m_model_plugin->q();
//     ukf_t::m_Ea.block(2*ukf_t::n_variables, 2*ukf_t::n_variables, ukf_t::n_measurements, ukf_t::n_measurements) = ukf_t::m_model_plugin->r();

//     // Generate sigma points for the current augmented state vector.
//     // Calculate sqrt((n+lambda)*Ea).
//     ukf_t::m_Ea *= (static_cast<double>(ukf_t::n_variables) + lambda);
//     // Use Cholesky decomposition to get the square root.
//     // Put square root in place of m_Ea since it's not longer needed.
//     ukf_t::m_llt.compute(ukf_t::m_Ea);
//     ukf_t::m_Ea = ukf_t::m_llt.matrixL();
//     // Set first column of the sigma matrix to the variable/mean vector.
//     ukf_t::m_X.col(0) = ukf_t::v_xa;
//     // Calculate and set reminaing sigma matrix columns.
//     for(j = 0; j < ukf_t::n_augmented; ++j)
//     {
//         ukf_t::m_X.col(j+1) = ukf_t::v_xa + ukf_t::m_Ea.col(j);
//         ukf_t::m_X.col(j+1+ukf_t::n_augmented) = ukf_t::v_xa - ukf_t::m_Ea.col(j);
//     }

//     // Predict new state and covariance.
//     // First, transition sigma points through the model plugin.
//     // Iterate over each column in the sigma point matrix.
//     for(j = 0; j < ukf_t::n_sigma; ++j)
//     {
//         // Populate the interface vector for prior state.
//         // The interface vector is the variable portion of the sigma column + the process noise portion of the sigma column.
//         ukf_t::i_xp = ukf_t::m_X.block(0, j, ukf_t::n_variables, 1) + ukf_t::m_X.block(ukf_t::n_variables, j, ukf_t::n_variables, 1);
//         // Reset interface vector for new state as a precaution.
//         ukf_t::i_x.setZero();
//         // Run model plugin's state transition method.
//         ukf_t::m_model_plugin->state_transition(ukf_t::i_xp, ukf_t::i_x);
//         // Capture transitioned state vector back into X (X not needed after this and need to calculate variance, so can't inline averaging)
//         ukf_t::m_X.block(0, j, ukf_t::n_variables, 1) = ukf_t::i_x;
//     }
//     // Calculate the predicted state as the mean of the transitioned sigma test points for each variable.
//     // Using the weight vector as a weighted average across each row of m_X is just matrix multiplication m_X * w
//     // This can be done over the prior state to change it to the predicted state.
//     ukf_t::v_x.noalias() = ukf_t::m_X.block(0, 0, ukf_t::n_variables, ukf_t::n_sigma) * ukf_t::v_w;
//     // Calculate the predicted covariance from the predicted state.
//     // This needs to calculate sum of w*(X-x)(X-x)' for each column in the sigma matrix, which is basically (X-x)*m_w*(X-x)'.
//     // This can be done over the prioer covariance to change it to the predicted covariance.
//     ukf_t::t_ns.noalias() = ukf_t::m_X.block(0, 0, ukf_t::n_variables, ukf_t::n_sigma) * ukf_t::m_w;
//     ukf_t::m_E.noalias() = ukf_t::t_ns * ukf_t::m_X.block(0, 0, ukf_t::n_variables, ukf_t::n_sigma).transpose();
// }