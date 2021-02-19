#include <kalman_filter/ukf.hpp>

using namespace kalman_filter;

// CONSTRUCTORS
ukf_t::ukf_t(uint32_t n_variables, uint32_t n_observers)
    : base_t(n_variables, n_observers)
{
    // Calculate number of sigma points.
    ukf_t::n_s = 1 + 2*ukf_t::n_x;

    // Allocate weight vectors.
    ukf_t::wm.setZero(ukf_t::n_s);
    ukf_t::wc.setZero(ukf_t::n_s);

    // Allocate sigma matrices
    ukf_t::X.setZero(ukf_t::n_x, ukf_t::n_s);
    ukf_t::Z.setZero(ukf_t::n_z, ukf_t::n_s);

    // Allocate interface components.
    ukf_t::i_xp.setZero(ukf_t::n_x);
    ukf_t::i_x.setZero(ukf_t::n_x);
    ukf_t::i_z.setZero(ukf_t::n_z);

    // Allocate temporaries.
    ukf_t::t_zz.setZero(ukf_t::n_z, ukf_t::n_z);
    ukf_t::t_xs.setZero(ukf_t::n_x, ukf_t::n_s);
    ukf_t::t_zs.setZero(ukf_t::n_z, ukf_t::n_s);

    // Set default parameters.
    ukf_t::alpha = 0.001;
    ukf_t::kappa = 3 - ukf_t::n_x;
    ukf_t::beta = 2;
}

// FILTER METHODS
void ukf_t::iterate()
{
    // ---------- STEP 1: PREPARATION ----------

    // Calculate lambda for this iteration (user can change parameters between iterations)
    double lambda = ukf_t::alpha * ukf_t::alpha * (static_cast<double>(ukf_t::n_x) + ukf_t::kappa) - static_cast<double>(ukf_t::n_x);

    // Calculate weight vectors for mean and covariance averaging.
    // Set mean recovery weight vector.
    ukf_t::wm.fill(1.0 / (2.0 * (static_cast<double>(ukf_t::n_x) + lambda)));
    ukf_t::wm(0) *= 2.0 * lambda;
    // Copy wc from wm and update first element.
    ukf_t::wc = ukf_t::wm;
    ukf_t::wc(0) += (1.0 - ukf_t::alpha*ukf_t::alpha + ukf_t::beta);

    // ---------- STEP 2: PREDICT ----------

    // Populate previous state sigma matrix
    // Calculate square root of P using Cholseky Decomposition
    ukf_t::llt.compute(ukf_t::P);
    // Check if calculation succeeded (positive semi definite)
    if(ukf_t::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix P is not positive semi definite");
    }
    // Reset first column of X.
    ukf_t::X.col(0).setZero();
    // Fill X with +sqrt(P)
    ukf_t::X.block(0,1,ukf_t::n_x,ukf_t::n_x) = ukf_t::llt.matrixL();
    // Fill X with -sqrt(P)
    ukf_t::X.block(0,1+ukf_t::n_x,ukf_t::n_x,ukf_t::n_x) = -1.0 * ukf_t::X.block(0,1,ukf_t::n_x,ukf_t::n_x);
    // Apply sqrt(n+lambda) to entire matrix.
    ukf_t::X *= std::sqrt(static_cast<double>(ukf_t::n_x) + lambda);
    // Add mean to entire matrix.
    ukf_t::X += ukf_t::x.replicate(1,ukf_t::n_s);

    // Pass previous X through state transition function.
    for(uint32_t s = 0; s < ukf_t::n_s; ++s)
    {
        // Populate interface vector.
        ukf_t::i_xp = ukf_t::X.col(s);
        ukf_t::i_x.setZero();
        // Evaluate state transition.
        state_transition(ukf_t::i_xp, ukf_t::i_x);
        // Store result back in X.
        ukf_t::X.col(s) = ukf_t::i_x;
    }

    // Calculate predicted state mean.
    ukf_t::x.noalias() = ukf_t::X * ukf_t::wm;

    // Calculate predicted state covariance.
    ukf_t::X -= ukf_t::x.replicate(1, ukf_t::n_s);
    ukf_t::t_xs.noalias() = ukf_t::X * ukf_t::wc.asDiagonal();
    ukf_t::P.noalias() = ukf_t::t_xs * ukf_t::X.transpose();
    ukf_t::P += ukf_t::Q;

    // ---------- STEP 3: UPDATE ----------

    // Get number of observations made.
    uint32_t n_o = ukf_t::m_observations.size();
    // Check if any observations have been made.
    if(n_o == 0)
    {
        // No new observations. No need to update.
        return;
    }

    // Populate predicted state sigma matrix.
    // Calculate square root of P using Cholseky Decomposition
    ukf_t::llt.compute(ukf_t::P);
    // Check if calculation succeeded (positive semi definite)
    if(ukf_t::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix P is not positive semi definite");
    }
    // Reset first column of X.
    ukf_t::X.col(0).setZero();
    // Fill X with +sqrt(P)
    ukf_t::X.block(0,1,ukf_t::n_x,ukf_t::n_x) = ukf_t::llt.matrixL();
    // Fill X with -sqrt(P)
    ukf_t::X.block(0,1+ukf_t::n_x,ukf_t::n_x,ukf_t::n_x) = -1.0 * ukf_t::X.block(0,1,ukf_t::n_x,ukf_t::n_x);
    // Apply sqrt(n+lambda) to entire matrix.
    ukf_t::X *= std::sqrt(static_cast<double>(ukf_t::n_x) + lambda);
    // Add mean to entire matrix.
    ukf_t::X += ukf_t::x.replicate(1,ukf_t::n_s);

    // Pass predicted X through state transition function.
    for(uint32_t s = 0; s < ukf_t::n_s; ++s)
    {
        // Populate interface vector.
        ukf_t::i_x = ukf_t::X.col(s);
        ukf_t::i_z.setZero();
        // Evaluate state transition.
        observation(ukf_t::i_x, ukf_t::i_z);
        // Store result back in X.
        ukf_t::Z.col(s) = ukf_t::i_z;
    }

    // Calculate predicted observation mean.
    ukf_t::z.noalias() = ukf_t::Z * ukf_t::wm;

    // Calculate predicted observation covariance.
    ukf_t::Z -= ukf_t::z.replicate(1, ukf_t::n_s);
    ukf_t::t_zs.noalias() = ukf_t::Z * ukf_t::wc.asDiagonal();
    ukf_t::S.noalias() = ukf_t::t_zs * ukf_t::Z.transpose();
    ukf_t::S += ukf_t::R;

    // Calculate predicted state/observation covariance.
    ukf_t::X -= ukf_t::x.replicate(1, ukf_t::n_s);
    ukf_t::t_xs.noalias() = ukf_t::X * ukf_t::wc.asDiagonal();
    ukf_t::C.noalias() = ukf_t::t_xs * ukf_t::Z.transpose();

    // Calculate inverse of predicted observation covariance.
    ukf_t::t_zz = ukf_t::S.inverse();

    // Using number of observations, create masked versions of S and S_i.
    Eigen::MatrixXd S_m(n_o, n_o);
    Eigen::MatrixXd Si_m(ukf_t::n_z, n_o);
    // Iterate over z indices.
    uint32_t m_i = 0;
    uint32_t m_j = 0;
    // Iterate column first.
    for(auto j = ukf_t::m_observations.begin(); j != ukf_t::m_observations.end(); ++j)
    {
        // Iterate over rows to populate O_m.
        for(auto i = ukf_t::m_observations.begin(); i != ukf_t::m_observations.end(); ++i)
        {
            // Copy the selected O element into O_m.
            S_m(m_i++, m_j) = ukf_t::S(i->first, j->first);
        }
        m_i = 0;

        // Copy the selected Oi column into Oi_m.
        Si_m.col(m_j++) = ukf_t::t_zz.col(j->first);
    }
    
    // Calculate Kalman gain (masked by n observations).
    Eigen::MatrixXd K_m(ukf_t::n_x,n_o);
    K_m.noalias() = ukf_t::C * Si_m;

    // Create masked version of za-z.
    Eigen::VectorXd zd_m(n_o);
    m_i = 0;
    for(auto observation = ukf_t::m_observations.begin(); observation != ukf_t::m_observations.end(); ++observation)
    {
        zd_m(m_i++) = observation->second - ukf_t::z(observation->first);
    }

    // Update state.
    ukf_t::x.noalias() += K_m * zd_m;

    // Update covariance.
    // NOTE: Just use internal temporary since it's masked size.
    ukf_t::P.noalias() -= K_m * S_m * K_m.transpose();

    // Reset observations.
    ukf_t::m_observations.clear();
}