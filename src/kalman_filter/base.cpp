#include <kalman_filter/base.hpp>

using namespace kalman_filter;

// CONSTRUCTORS
base_t::base_t(uint32_t n_variables, uint32_t n_observers)
{
    // Store dimension sizes.
    base_t::n_x = n_variables;
    base_t::n_z = n_observers;

    // Allocate prediction components.
    base_t::x.setZero(base_t::n_x);
    base_t::P.setIdentity(base_t::n_x, base_t::n_x);
    base_t::Q.setIdentity(base_t::n_x, base_t::n_x);

    // Allocate update components.
    base_t::R.setIdentity(base_t::n_z, base_t::n_z);
    base_t::z.setZero(base_t::n_z);
    base_t::S.setZero(base_t::n_z, base_t::n_z);
    base_t::C.setZero(base_t::n_x, base_t::n_z);
}

// FILTER METHODS
void base_t::initialize_state(const Eigen::VectorXd& initial_state, const Eigen::MatrixXd& initial_covariance)
{
    // Verify initial state size.
    if(initial_state.size() != base_t::n_x)
    {
        throw std::runtime_error("failed to initialize state vector (initial state provided has incorrect size)");
    }

    // Verify initial covariance size.
    if(initial_covariance.rows() != base_t::n_x || initial_covariance.cols() != base_t::n_x)
    {
        throw std::runtime_error("failed to initialize state covariance (initial covariance provided has incorrect size)");
    }

    // Copy initial state and covariance.
    base_t::x = initial_state;
    base_t::P = initial_covariance;
}
void base_t::new_observation(uint32_t observer_index, double_t observation)
{
    // Store observation in the observations map.
    // NOTE: This adds or replaces the observation at the specified observer index.
    base_t::m_observations[observer_index] = observation;
}

// ACCESS
uint32_t base_t::n_variables() const
{
    return base_t::n_x;
}
uint32_t base_t::n_observers() const
{
    return base_t::n_z;
}
const Eigen::VectorXd& base_t::state() const
{
    return base_t::x;
}
const Eigen::MatrixXd& base_t::covariance() const
{
    return base_t::P;
}