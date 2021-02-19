#include <kalman_filter/ukfa.hpp>

using namespace kalman_filter;

// CONSTRUCTORS
ukfa_t::ukfa_t(uint32_t n_variables, uint32_t n_observers)
    : base_t(n_variables, n_observers)
{
    // Store augmented dimension sizes.
    ukfa_t::n_a = ukfa_t::n_x + ukfa_t::n_x + ukfa_t::n_z;
    ukfa_t::n_s = 1 + 2*ukfa_t::n_a;

    // Allocate weight vectors.
    ukfa_t::wm.setZero(ukfa_t::n_s);
    ukfa_t::wc.setZero(ukfa_t::n_s);

    // Allocate prediction components.
    ukfa_t::Xp.setZero(ukfa_t::n_x, ukfa_t::n_x);
    ukfa_t::Xq.setZero(ukfa_t::n_x, ukfa_t::n_x);
    ukfa_t::X.setZero(ukfa_t::n_x, ukfa_t::n_s);
    ukfa_t::dX.setZero(ukfa_t::n_x, ukfa_t::n_s);

    // Allocate update components.
    ukfa_t::Xr.setZero(ukfa_t::n_z, ukfa_t::n_z);
    ukfa_t::Z.setZero(ukfa_t::n_z, ukfa_t::n_s);

    // Allocate interface components.
    ukfa_t::i_xp.setZero(ukfa_t::n_x);
    ukfa_t::i_x.setZero(ukfa_t::n_x);
    ukfa_t::i_q.setZero(ukfa_t::n_x);
    ukfa_t::i_r.setZero(ukfa_t::n_z);
    ukfa_t::i_z.setZero(ukfa_t::n_z);

    // Allocate temporaries.
    ukfa_t::t_zz.setZero(ukfa_t::n_z, ukfa_t::n_z);
    ukfa_t::t_xs.setZero(ukfa_t::n_x, ukfa_t::n_s);
    ukfa_t::t_zs.setZero(ukfa_t::n_z, ukfa_t::n_s);

    // Set default parameters.
    ukfa_t::alpha = 0.001;
    ukfa_t::kappa = 3 - ukfa_t::n_x;
    ukfa_t::beta = 2;
}

// FILTER METHODS
void ukfa_t::iterate()
{
    // ---------- STEP 1: PREPARATION ----------
    // Calculate lambda for this iteration (user can change parameters between iterations)
    double lambda = ukfa_t::alpha * ukfa_t::alpha * (static_cast<double>(ukfa_t::n_a) + ukfa_t::kappa) - static_cast<double>(ukfa_t::n_a);

    // Calculate weight vectors for mean and covariance averaging.
    // Set mean recovery weight vector.
    ukfa_t::wm.fill(1.0 / (2.0 * (static_cast<double>(ukfa_t::n_a) + lambda)));
    ukfa_t::wm(0) *= 2.0 * lambda;
    // Copy wc from wm and update first element.
    ukfa_t::wc = ukfa_t::wm;
    ukfa_t::wc(0) += (1.0 - ukfa_t::alpha*ukfa_t::alpha + ukfa_t::beta);

    // ---------- STEP 2: PREDICT ----------
    // Calculate sigma matrix.
    // NOTE: This implementation segments out the input sigma matrix for efficiency:
    // [u u+y*sqrt(P) u-y*sqrt(P) 0           0           0           0          ]
    // [0 0           0           u+y(sqrt(Q) u-y*sqrt(Q) 0           0          ]
    // [0 0           0           0           0           u+y*sqrt(R) u-y*sqrt(R)]
    // u is stored in x
    // y*sqrt(P) stored in Xp
    // y*sqrt(Q) stored in Xq
    // y*sqrt(R) stored in Xr.

    // Calculate square root of P using Cholseky Decomposition
    ukfa_t::llt.compute(ukfa_t::P);
    // Check if calculation succeeded (positive semi definite)
    if(ukfa_t::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix P is not positive semi definite");
    }
    // Fill +sqrt(P) block of Xp.
    ukfa_t::Xp = ukfa_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukfa_t::Xp *= std::sqrt(static_cast<double>(ukfa_t::n_a) + lambda);

    // Calculate square root of Q using Cholseky Decomposition.
    ukfa_t::llt.compute(ukfa_t::Q);
    // Check if calculation succeeded (positive semi definite)
    if(ukfa_t::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix Q is not positive semi definite");
    }
    // Fill +sqrt(Q) block of Xq.
    ukfa_t::Xq = ukfa_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukfa_t::Xq *= std::sqrt(static_cast<double>(ukfa_t::n_a) + lambda);

    // Calculate square root of R using Cholseky Decomposition.
    ukfa_t::llt.compute(ukfa_t::R);
    // Check if calculation succeeded (positive semi definite)
    if(ukfa_t::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix R is not positive semi definite");
    }
    // Fill +sqrt(R) block of Xr.
    ukfa_t::Xr = ukfa_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukfa_t::Xr *= std::sqrt(static_cast<double>(ukfa_t::n_a) + lambda);

    // Calculate X by passing sigma points through the transition function.

    // Create sigma column index.
    uint32_t s = 0;

    // Pass first set of sigma points, which is just the mean.
    // Populate interface vectors.
    ukfa_t::i_xp = ukfa_t::x;
    ukfa_t::i_q.setZero(ukfa_t::n_x);
    ukfa_t::i_x.setZero(ukfa_t::n_x);
    // Run transition function.
    state_transition(ukfa_t::i_xp, ukfa_t::i_q, ukfa_t::i_x);
    // Capture output into X.
    ukfa_t::X.col(s++) = ukfa_t::i_x;

    // Pass second set of sigma points, which injects Xp.
    for(uint32_t j = 0; j < ukfa_t::n_x; ++j)
    {
        // mean PLUS y*sqrt(P)
        // Populate interface vectors.
        ukfa_t::i_xp = ukfa_t::x + ukfa_t::Xp.col(j);
        ukfa_t::i_q.setZero(ukfa_t::n_x);
        ukfa_t::i_x.setZero(ukfa_t::n_x);
        // Run transition function.
        state_transition(ukfa_t::i_xp, ukfa_t::i_q, ukfa_t::i_x);
        // Capture output into X.
        ukfa_t::X.col(s++) = ukfa_t::i_x;
    }
    for(uint32_t j = 0; j < ukfa_t::n_x; ++j)
    {
        // mean MINUS y*sqrt(P)
        // Populate interface vectors.
        ukfa_t::i_xp = ukfa_t::x - ukfa_t::Xp.col(j);
        ukfa_t::i_q.setZero(ukfa_t::n_x);
        ukfa_t::i_x.setZero(ukfa_t::n_x);
        // Run transition function.
        state_transition(ukfa_t::i_xp, ukfa_t::i_q, ukfa_t::i_x);
        // Capture output into X.
        ukfa_t::X.col(s++) = ukfa_t::i_x;
    }

    // Pass third set of sigma points, which injects Xq.
    for(uint32_t j = 0; j < ukfa_t::n_x; ++j)
    {
        // mean PLUS y*sqrt(Q)
        // Populate interface vectors.
        ukfa_t::i_xp = ukfa_t::x;
        ukfa_t::i_q = ukfa_t::Xq.col(j);
        ukfa_t::i_x.setZero(ukfa_t::n_x);
        // Run transition function.
        state_transition(ukfa_t::i_xp, ukfa_t::i_q, ukfa_t::i_x);
        // Capture output into X.
        ukfa_t::X.col(s++) = ukfa_t::i_x;
    }
    for(uint32_t j = 0; j < ukfa_t::n_x; ++j)
    {  
        // mean MINUS y*sqrt(Q)
        // Populate interface vectors.
        ukfa_t::i_xp = ukfa_t::x;
        ukfa_t::i_q = -ukfa_t::Xq.col(j);
        ukfa_t::i_x.setZero(ukfa_t::n_x);
        // Run transition function.
        state_transition(ukfa_t::i_xp, ukfa_t::i_q, ukfa_t::i_x);
        // Capture output into X.
        ukfa_t::X.col(s++) = ukfa_t::i_x;
    }

    // Pass fourth set of sigma points, which injects Xr.
    // R has no effect on the transition function, so the output sigma matrix
    // just has extra copies of the mean at the end.
    for(;s < ukfa_t::n_s; ++s)
    {
        ukfa_t::X.col(s) = ukfa_t::X.col(0);
    }

    // Calculate predicted state mean and covariance.
    
    // Predicted state mean is a weighted average: sum(wm.*X) over all sigma points.
    // Can be calculated via matrix multiplication with wm vector.
    ukfa_t::x.noalias() = ukfa_t::X * ukfa_t::wm;

    // Predicted state covariance is a weighted average: sum(wc.*(X-x)(X-x)') over all sigma points.
    // This can be done more efficiently (speed & code) using (X-x)*wc*(X-x)', where wc is formed into a diagonal matrix.
    ukfa_t::dX = ukfa_t::X - ukfa_t::x.replicate(1, ukfa_t::n_s);
    ukfa_t::t_xs.noalias() = ukfa_t::dX * ukfa_t::wc.asDiagonal();
    ukfa_t::P.noalias() = ukfa_t::t_xs * ukfa_t::dX.transpose();

    // ---------- STEP 3: UPDATE ----------
    // Get number of observations made.
    uint32_t n_o = ukfa_t::m_observations.size();
    // Check if any observations have been made.
    if(n_o == 0)
    {
        // No new observations. No need to update.
        return;
    }

    // Calculate Z by passing calculated X and Sr.

    // Pass the x/Xp/Xq portion of X through.
    for(s = 0; s < 1 + 4 * ukfa_t::n_x; ++s)
    {
        // Populate interface vectors.
        ukfa_t::i_x = ukfa_t::X.col(s);
        ukfa_t::i_r.setZero(ukfa_t::n_z);
        ukfa_t::i_z.setZero(ukfa_t::n_z);
        // Run observation function.
        observation(ukfa_t::i_x, ukfa_t::i_r, ukfa_t::i_z);
        // Capture output into Z.
        ukfa_t::Z.col(s) = ukfa_t::i_z;
    }

    // Pass Sr through on top of the back of X.
    for(uint32_t j = 0; j < ukfa_t::n_z; ++j)
    {
        // mean PLUS y*sqrt(R)
        // Populate interface vectors.
        ukfa_t::i_x = ukfa_t::X.col(s);
        ukfa_t::i_r = ukfa_t::Xr.col(j);
        ukfa_t::i_z.setZero(ukfa_t::n_z);
        // Run observation function.
        observation(ukfa_t::i_x, ukfa_t::i_r, ukfa_t::i_z);
        // Capture output into Z.
        ukfa_t::Z.col(s++) = ukfa_t::i_z;
    }
    for(uint32_t j = 0; j < ukfa_t::n_z; ++j)
    {
        // mean MINUS y*sqrt(R)
        // Populate interface vectors.
        ukfa_t::i_x = ukfa_t::X.col(s);
        ukfa_t::i_r = -ukfa_t::Xr.col(j);
        ukfa_t::i_z.setZero(ukfa_t::n_z);
        // Run observation function.
        observation(ukfa_t::i_x, ukfa_t::i_r, ukfa_t::i_z);
        // Capture output into Z.
        ukfa_t::Z.col(s++) = ukfa_t::i_z;
    }

    // Calculate predicted observation mean and covariance, as well as cross covariance.
    
    // Predicted observation mean is a weighted average: sum(wm.*Z) over all sigma points.
    // Can be calculated via matrix multiplication with wm vector.
    ukfa_t::z.noalias() = ukfa_t::Z * ukfa_t::wm;

    // Predicted observation covariance is a weighted average: sum(wc.*(Z-z)(Z-z)') over all sigma points.
    // This can be done more efficiently (speed & code) using (Z-z)*wc*(Z-z)', where wc is formed into a diagonal matrix.
    // Calculate Z-z in place on Z as it's not needed afterwards.
    ukfa_t::Z -= ukfa_t::z.replicate(1, ukfa_t::n_s);
    ukfa_t::t_zs.noalias() = ukfa_t::Z * ukfa_t::wc.asDiagonal();
    ukfa_t::S.noalias() = ukfa_t::t_zs * ukfa_t::Z.transpose();

    // Predicted state/observation cross covariance is a weighted average: sum(wc.*(X-x)(Z-z)') over all sigma points.
    // This can be done more efficiently (speed & code) using (X-x)*wc*(Z-z)', where wc is formed into a diagonal matrix.
    // Recall that (X-x)*wc is currently stored in ukfa_t::t_xs, and Z-z is stored in Z.
    ukfa_t::C.noalias() = ukfa_t::t_xs * ukfa_t::Z.transpose();

    // Calculate inverse of predicted observation covariance.
    ukfa_t::t_zz = ukfa_t::S.inverse();

    // Using number of observations, create masked versions of S and S_i.
    Eigen::MatrixXd S_m(n_o, n_o);
    Eigen::MatrixXd Si_m(ukfa_t::n_z, n_o);
    // Iterate over z indices.
    uint32_t m_i = 0;
    uint32_t m_j = 0;
    // Iterate column first.
    for(auto j = ukfa_t::m_observations.begin(); j != ukfa_t::m_observations.end(); ++j)
    {
        // Iterate over rows to populate O_m.
        for(auto i = ukfa_t::m_observations.begin(); i != ukfa_t::m_observations.end(); ++i)
        {
            // Copy the selected O element into O_m.
            S_m(m_i++, m_j) = ukfa_t::S(i->first, j->first);
        }
        m_i = 0;

        // Copy the selected Oi column into Oi_m.
        Si_m.col(m_j++) = ukfa_t::t_zz.col(j->first);
    }
    
    // Calculate Kalman gain (masked by n observations).
    Eigen::MatrixXd K_m(ukfa_t::n_x,n_o);
    K_m.noalias() = ukfa_t::C * Si_m;

    // Create masked version of za-z.
    Eigen::VectorXd zd_m(n_o);
    m_i = 0;
    for(auto observation = ukfa_t::m_observations.begin(); observation != ukfa_t::m_observations.end(); ++observation)
    {
        zd_m(m_i++) = observation->second - ukfa_t::z(observation->first);
    }

    // Update state.
    ukfa_t::x.noalias() += K_m * zd_m;

    // Update covariance.
    // NOTE: Just use internal temporary since it's masked size.
    ukfa_t::P.noalias() -= K_m * S_m * K_m.transpose();

    // Reset observations.
    ukfa_t::m_observations.clear();
}