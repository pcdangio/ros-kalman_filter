#include <kalman_filter/ukf.hpp>

using namespace kalman_filter;

// CONSTRUCTORS
ukf_t::ukf_t(uint32_t n_variables, uint32_t n_observers)
{
    // Store dimension sizes.
    ukf_t::n_x = n_variables;
    ukf_t::n_z = n_observers;
    ukf_t::n_a = ukf_t::n_x + ukf_t::n_x + ukf_t::n_z;
    ukf_t::n_s = 1 + 2*ukf_t::n_a;

    // Allocate weight vectors.
    ukf_t::wm.setZero(ukf_t::n_s);
    ukf_t::wc.setZero(ukf_t::n_s);

    // Allocate prediction components.
    ukf_t::x.setZero(ukf_t::n_x);
    ukf_t::P.setIdentity(ukf_t::n_x, ukf_t::n_x);
    ukf_t::Q.setIdentity(ukf_t::n_x, ukf_t::n_x);
    ukf_t::Xp.setZero(ukf_t::n_x, ukf_t::n_x);
    ukf_t::Xq.setZero(ukf_t::n_x, ukf_t::n_x);
    ukf_t::X.setZero(ukf_t::n_x, ukf_t::n_s);

    // Allocate update components.
    ukf_t::R.setIdentity(ukf_t::n_z, ukf_t::n_z);
    ukf_t::Xr.setZero(ukf_t::n_z, ukf_t::n_z);
    ukf_t::Z.setZero(ukf_t::n_z, ukf_t::n_s);
    ukf_t::z.setZero(ukf_t::n_z);
    ukf_t::S.setZero(ukf_t::n_z, ukf_t::n_z);
    ukf_t::C.setZero(ukf_t::n_x, ukf_t::n_z);

    // Allocate interface components.
    ukf_t::i_xp.setZero(ukf_t::n_x);
    ukf_t::i_x.setZero(ukf_t::n_x);
    ukf_t::i_q.setZero(ukf_t::n_x);
    ukf_t::i_r.setZero(ukf_t::n_z);
    ukf_t::i_z.setZero(ukf_t::n_z);

    // Allocate temporaries.
    ukf_t::t_x.setZero(ukf_t::n_x);
    ukf_t::t_xx.setZero(ukf_t::n_x, ukf_t::n_x);
    ukf_t::t_z.setZero(ukf_t::n_z);
    ukf_t::t_zz.setZero(ukf_t::n_z, ukf_t::n_z);
    ukf_t::t_xz.setZero(ukf_t::n_x, ukf_t::n_z);

    // Set default parameters.
    ukf_t::alpha = 0.001;
    ukf_t::kappa = 3 - ukf_t::n_x;
    ukf_t::beta = 2;
}

// FILTER METHODS
void ukf_t::initialize_state(const Eigen::VectorXd& initial_state, const Eigen::MatrixXd& initial_covariance)
{
    // Verify initial state size.
    if(initial_state.size() != ukf_t::n_x)
    {
        throw std::runtime_error("failed to initialize state vector (initial state provided has incorrect size)");
    }

    // Verify initial covariance size.
    if(initial_covariance.rows() != ukf_t::n_x || initial_covariance.cols() != ukf_t::n_x)
    {
        throw std::runtime_error("failed to initialize state covariance (initial covariance provided has incorrect size)");
    }

    // Copy initial state and covariance.
    ukf_t::x = initial_state;
    ukf_t::P = initial_covariance;
}
void ukf_t::iterate()
{
    // Calculate lambda for this iteration (user can change parameters between iterations)
    double lambda = ukf_t::alpha * ukf_t::alpha * (static_cast<double>(ukf_t::n_a) + ukf_t::kappa) - static_cast<double>(ukf_t::n_a);

    // Calculate weight vectors for mean and covariance averaging.
    // Set mean recovery weight vector.
    ukf_t::wm.fill(1.0 / (2.0 * (static_cast<double>(ukf_t::n_a) + lambda)));
    ukf_t::wm(0) *= 2.0 * lambda;
    // Copy wc from wm and update first element.
    ukf_t::wc = ukf_t::wm;
    ukf_t::wc(0) += (1.0 - ukf_t::alpha*ukf_t::alpha + ukf_t::beta);

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
    ukf_t::llt.compute(ukf_t::P);
    // Check if calculation succeeded (positive semi definite)
    if(ukf_t::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix P is not positive semi definite");
    }
    // Fill +sqrt(P) block of Xp.
    ukf_t::Xp = ukf_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukf_t::Xp *= std::sqrt(static_cast<double>(ukf_t::n_a) + lambda);

    // Calculate square root of Q using Cholseky Decomposition.
    ukf_t::llt.compute(ukf_t::Q);
    // Check if calculation succeeded (positive semi definite)
    if(ukf_t::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix Q is not positive semi definite");
    }
    // Fill +sqrt(Q) block of Xq.
    ukf_t::Xq = ukf_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukf_t::Xq *= std::sqrt(static_cast<double>(ukf_t::n_a) + lambda);

    // Calculate square root of R using Cholseky Decomposition.
    ukf_t::llt.compute(ukf_t::R);
    // Check if calculation succeeded (positive semi definite)
    if(ukf_t::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix R is not positive semi definite");
    }
    // Fill +sqrt(R) block of Xr.
    ukf_t::Xr = ukf_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukf_t::Xr *= std::sqrt(static_cast<double>(ukf_t::n_a) + lambda);

    // Calculate X by passing sigma points through the transition function.

    // Create sigma column index.
    uint32_t s = 0;

    // Pass first set of sigma points, which is just the mean.
    // Populate interface vectors.
    ukf_t::i_xp = ukf_t::x;
    ukf_t::i_q.setZero(ukf_t::n_x);
    ukf_t::i_x.setZero(ukf_t::n_x);
    // Run transition function.
    state_transition(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
    // Capture output into X.
    ukf_t::X.col(s++) = ukf_t::i_x;

    // Pass second set of sigma points, which injects Xp.
    for(uint32_t j = 0; j < ukf_t::n_x; ++j)
    {
        // mean PLUS y*sqrt(P)
        // Populate interface vectors.
        ukf_t::i_xp = ukf_t::x + ukf_t::Xp.col(j);
        ukf_t::i_q.setZero(ukf_t::n_x);
        ukf_t::i_x.setZero(ukf_t::n_x);
        // Run transition function.
        state_transition(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.col(s++) = ukf_t::i_x;
    }
    for(uint32_t j = 0; j < ukf_t::n_x; ++j)
    {
        // mean MINUS y*sqrt(P)
        // Populate interface vectors.
        ukf_t::i_xp = ukf_t::x - ukf_t::Xp.col(j);
        ukf_t::i_q.setZero(ukf_t::n_x);
        ukf_t::i_x.setZero(ukf_t::n_x);
        // Run transition function.
        state_transition(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.col(s++) = ukf_t::i_x;
    }

    // Pass third set of sigma points, which injects Xq.
    for(uint32_t j = 0; j < ukf_t::n_x; ++j)
    {
        // mean PLUS y*sqrt(Q)
        // Populate interface vectors.
        ukf_t::i_xp = ukf_t::x;
        ukf_t::i_q = ukf_t::Xq.col(j);
        ukf_t::i_x.setZero(ukf_t::n_x);
        // Run transition function.
        state_transition(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.col(s++) = ukf_t::i_x;
    }
    for(uint32_t j = 0; j < ukf_t::n_x; ++j)
    {  
        // mean MINUS y*sqrt(Q)
        // Populate interface vectors.
        ukf_t::i_xp = ukf_t::x;
        ukf_t::i_q = -ukf_t::Xq.col(j);
        ukf_t::i_x.setZero(ukf_t::n_x);
        // Run transition function.
        state_transition(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.col(s++) = ukf_t::i_x;
    }

    // Pass fourth set of sigma points, which injects Xr.
    // R has no effect on the transition function, so the output sigma matrix
    // just has extra copies of the mean at the end.
    for(;s < ukf_t::n_s; ++s)
    {
        ukf_t::X.col(s) = ukf_t::X.col(0);
    }



    // Calculate predicted state mean and covariance.
    
    // Predicted state mean is a weighted average: sum(wm.*X) over all sigma points.
    // Can be calculated via matrix multiplication with wm vector.
    ukf_t::x.noalias() = ukf_t::X * ukf_t::wm;

    // Predicted state covariance is a weighted average: sum(wc.*(X-x)(X-x)') over all sigma points.
    ukf_t::P.setZero();
    for(s = 0; s < ukf_t::n_s; ++s)
    {
        // Calculate X-x at this sigma column.
        ukf_t::t_x = ukf_t::X.col(s) - ukf_t::x;
        // Calculate outer product.
        ukf_t::t_xx.noalias() = ukf_t::t_x * ukf_t::t_x.transpose();
        // Apply weight to outer product.
        ukf_t::t_xx *= ukf_t::wc(s);
        // Sum into predicted state covariance.
        ukf_t::P += ukf_t::t_xx;
    } 



    // UPDATE

    // Get number of observations made.
    uint32_t n_o = ukf_t::m_observations.size();

    // Check if any observations have been made.
    if(n_o == 0)
    {
        // No new observations. No need to update.
        return;
    }



    // Calculate Z by passing calculated X and Sr.

    // Pass the x/Xp/Xq portion of X through.
    for(s = 0; s < 1 + 4 * ukf_t::n_x; ++s)
    {
        // Populate interface vectors.
        ukf_t::i_x = ukf_t::X.col(s);
        ukf_t::i_r.setZero(ukf_t::n_z);
        ukf_t::i_z.setZero(ukf_t::n_z);
        // Run observation function.
        observation(ukf_t::i_x, ukf_t::i_r, ukf_t::i_z);
        // Capture output into Z.
        ukf_t::Z.col(s) = ukf_t::i_z;
    }

    // Pass Sr through on top of the back of X.
    for(uint32_t j = 0; j < ukf_t::n_z; ++j)
    {
        // mean PLUS y*sqrt(R)
        // Populate interface vectors.
        ukf_t::i_x = ukf_t::X.col(s);
        ukf_t::i_r = ukf_t::Xr.col(j);
        ukf_t::i_z.setZero(ukf_t::n_z);
        // Run observation function.
        observation(ukf_t::i_x, ukf_t::i_r, ukf_t::i_z);
        // Capture output into Z.
        ukf_t::Z.col(s++) = ukf_t::i_z;
    }
    for(uint32_t j = 0; j < ukf_t::n_z; ++j)
    {
        // mean MINUS y*sqrt(R)
        // Populate interface vectors.
        ukf_t::i_x = ukf_t::X.col(s);
        ukf_t::i_r = -ukf_t::Xr.col(j);
        ukf_t::i_z.setZero(ukf_t::n_z);
        // Run observation function.
        observation(ukf_t::i_x, ukf_t::i_r, ukf_t::i_z);
        // Capture output into Z.
        ukf_t::Z.col(s++) = ukf_t::i_z;
    }



    // Calculate predicted observation mean and covariance, as well as cross covariance.
    
    // Predicted observation mean is a weighted average: sum(wm.*Z) over all sigma points.
    // Can be calculated via matrix multiplication with wm vector.
    ukf_t::z.noalias() = ukf_t::Z * ukf_t::wm;

    // Predicted observation covariance is a weighted average: sum(wc.*(Z-z)(Z-z)') over all sigma points.
    // Same for predicted state/observation cross covariance: sum(wc.*(X-x)(Z-z)')
    ukf_t::S.setZero();
    ukf_t::C.setZero();
    for(s = 0; s < ukf_t::n_s; ++s)
    {
        // Calculate Z-z and X-x at this sigma column.
        ukf_t::t_z = ukf_t::Z.col(s) - ukf_t::z;
        ukf_t::t_x = ukf_t::X.col(s) - ukf_t::x;
        // Calculate outer products.
        ukf_t::t_zz.noalias() = ukf_t::t_z * ukf_t::t_z.transpose();
        ukf_t::t_xz.noalias() = ukf_t::t_x * ukf_t::t_z.transpose();
        // Apply weight to outer products.
        ukf_t::t_zz *= ukf_t::wc(s);
        ukf_t::t_xz *= ukf_t::wc(s);
        // Sum into predicted covariances.
        ukf_t::S += ukf_t::t_zz;
        ukf_t::C += ukf_t::t_xz;
    }



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
    Eigen::MatrixXd K_m(n_x,n_o);
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
void ukf_t::new_observation(uint32_t observer_index, double_t observation)
{
    // Store observation in the observations map.
    // NOTE: This adds or replaces the observation at the specified observer index.
    ukf_t::m_observations[observer_index] = observation;
}

// ACCESS
uint32_t ukf_t::n_variables() const
{
    return ukf_t::n_x;
}
uint32_t ukf_t::n_observers() const
{
    return ukf_t::n_z;
}
const Eigen::VectorXd& ukf_t::state() const
{
    return ukf_t::x;
}
const Eigen::MatrixXd& ukf_t::covariance() const
{
    return ukf_t::P;
}