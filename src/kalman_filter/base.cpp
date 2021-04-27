#include <kalman_filter/base.hpp>

#include <fstream>

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

    // Allocate temporaries.
    base_t::t_zz.setZero(base_t::n_z, base_t::n_z);
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
    // Verify index exists.
    if(!(observer_index < base_t::n_z))
    {
        throw std::runtime_error("failed to add new observation (observer_index out of range)");
    }
    
    // Store observation in the observations map.
    // NOTE: This adds or replaces the observation at the specified observer index.
    base_t::m_observations[observer_index] = observation;
}
bool base_t::has_observations() const
{
    return !base_t::m_observations.empty();
}
void base_t::masked_kalman_update()
{
    // Get number of observations.
    uint32_t n_o = base_t::m_observations.size();

    // Calculate inverse of predicted observation covariance.
    base_t::t_zz = base_t::S.inverse();

    // Using number of observations, create masked versions of S and S_i.
    Eigen::MatrixXd S_m(n_o, n_o);
    Eigen::MatrixXd Si_m(base_t::n_z, n_o);
    // Iterate over z indices.
    uint32_t m_i = 0;
    uint32_t m_j = 0;
    // Iterate column first.
    for(auto j = base_t::m_observations.begin(); j != base_t::m_observations.end(); ++j)
    {
        // Iterate over rows to populate O_m.
        for(auto i = base_t::m_observations.begin(); i != base_t::m_observations.end(); ++i)
        {
            // Copy the selected O element into O_m.
            S_m(m_i++, m_j) = base_t::S(i->first, j->first);
        }
        m_i = 0;

        // Copy the selected Oi column into Oi_m.
        Si_m.col(m_j++) = base_t::t_zz.col(j->first);
    }
    
    // Calculate Kalman gain (masked by n observations).
    Eigen::MatrixXd K_m(base_t::n_x,n_o);
    K_m.noalias() = base_t::C * Si_m;

    // Create masked version of za-z.
    Eigen::VectorXd zd_m(n_o);
    m_i = 0;
    for(auto observation = base_t::m_observations.begin(); observation != base_t::m_observations.end(); ++observation)
    {
        zd_m(m_i++) = observation->second - base_t::z(observation->first);
    }

    // Update state.
    base_t::x.noalias() += K_m * zd_m;

    // Update covariance.
    // NOTE: Just use internal temporary since it's masked size.
    base_t::P.noalias() -= K_m * S_m * K_m.transpose();

    // Reset observations.
    base_t::m_observations.clear();
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
void base_t::modify_state(uint32_t index, double_t value)
{
    if(index >= base_t::n_x)
    {
        throw std::runtime_error("failed to modify state (state index out of range)");
    }
    
    base_t::x(index) = value;
}

// LOGGING
bool base_t::start_log(const std::string& log_file)
{
    // Stop any existing log.
    base_t::stop_log();

    // Open the file for writing.
    base_t::m_log_file.open(log_file.c_str());

    // Verify that the file opened correctly.
    if(base_t::m_log_file.fail())
    {
        // Close the stream and clear flags.
        base_t::m_log_file.close();
        base_t::m_log_file.clear();

        return false;
    }

    return true;
}
void base_t::stop_log()
{
    // Check if a log is running.
    if(base_t::m_log_file.is_open())
    {
        // Close the stream and reset flags.
        base_t::m_log_file.close();
        base_t::m_log_file.clear();
    }
}