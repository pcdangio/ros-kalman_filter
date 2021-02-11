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

    // Set up input sigma matrices.
    // NOTE: This implementation segments out the input sigma matrix for efficiency:
    // [u u+y*sqrt(P) u-y*sqrt(P) 0           0           0           0          ]
    // [0 0           0           u+y(sqrt(Q) u-y*sqrt(Q) 0           0          ]
    // [0 0           0           0           0           u+y*sqrt(R) u-y*sqrt(R)]
    // u is stored in x
    // y*sqrt(P) stored in Xp
    // y*sqrt(Q) stored in Xq
    // y*sqrt(R) stored in Xr.
    ukf_t::populate_sigma_component(ukf_t::n_a, lambda, ukf_t::P, ukf_t::Xp);
    ukf_t::populate_sigma_component(ukf_t::n_a, lambda, ukf_t::Q, ukf_t::Xq);
    ukf_t::populate_sigma_component(ukf_t::n_a, lambda, ukf_t::R, ukf_t::Xr);



    // Calculate X by passing x, Xp, and Xq through the transition function.

    // Pass first set of sigma points, which is just the mean.
    // Populate interface vectors.
    ukf_t::i_xp = ukf_t::x;
    ukf_t::i_q.setZero(ukf_t::n_x);
    ukf_t::i_x.setZero(ukf_t::n_x);
    // Run transition function.
    ukf_t::state_transition(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
    // Capture output into X.
    ukf_t::X.col(0) = ukf_t::i_x;

    // Pass second set of sigma points, which focuses on covariance P.
    for(uint32_t j = 0; j < ukf_t::Xp.cols(); ++j)
    {
        // mean PLUS y*sqrt(P)
        // Populate interface vectors.
        ukf_t::i_xp = ukf_t::x + ukf_t::Xp.col(j);
        ukf_t::i_q.setZero(ukf_t::n_x);
        ukf_t::i_x.setZero(ukf_t::n_x);
        // Run transition function.
        ukf_t::state_transition(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.col(1 + j) = ukf_t::i_x;
        
        // mean MINUS y*sqrt(P)
        // Populate interface vectors.
        ukf_t::i_xp = ukf_t::x - ukf_t::Xp.col(j);
        ukf_t::i_q.setZero(ukf_t::n_x);
        ukf_t::i_x.setZero(ukf_t::n_x);
        // Run transition function.
        ukf_t::state_transition(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.col(1 + ukf_t::n_x + j) = ukf_t::i_x;
    }

    // Pass third set of sigma points, which focuses on covariance Q.
    for(uint32_t j = 0; j < ukf_t::Xq.cols(); ++j)
    {
        // mean PLUS y*sqrt(Q)
        // Populate interface vectors.
        ukf_t::i_xp = ukf_t::x;
        ukf_t::i_q = ukf_t::Xq.col(j);
        ukf_t::i_x.setZero(ukf_t::n_x);
        // Run transition function.
        ukf_t::state_transition(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.col(1 + 2*ukf_t::n_x + j) = ukf_t::i_x;
        
        // mean MINUS y*sqrt(Q)
        // Populate interface vectors.
        ukf_t::i_xp = ukf_t::x;
        ukf_t::i_q = -ukf_t::Xq.col(j);
        ukf_t::i_x.setZero(ukf_t::n_x);
        // Run transition function.
        ukf_t::state_transition(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.col(1 + 3*ukf_t::n_x + j) = ukf_t::i_x;
    }



    // Calculate Z by passing X (x,Xp portion) and Xr.
    // NOTE: Do this with pure X so nonlinearities of X are passed directly into H(x).

    // Pass (x,Xp) portion of X through.
    for(uint32_t j = 0; j < 1 + 2 * ukf_t::n_x; ++j)
    {
        // Populate interface vectors.
        ukf_t::i_x = ukf_t::X.col(j);
        ukf_t::i_r.setZero(ukf_t::n_z);
        ukf_t::i_z.setZero(ukf_t::n_z);
        // Run observation function.
        ukf_t::observation(ukf_t::i_x, ukf_t::i_r, ukf_t::i_z);
        // Capture output into Z.
        ukf_t::Z.col(j) = ukf_t::i_z;
    }

    // Pass Xr through.
    for(uint32_t j = 0; j < ukf_t::Xr.cols(); ++j)
    {
        // mean PLUS y*sqrt(R)
        // Populate interface vectors.
        ukf_t::i_x = ukf_t::X.col(0);
        ukf_t::i_r = ukf_t::Xr.col(j);
        ukf_t::i_z.setZero(ukf_t::n_z);
        // Run observation function.
        ukf_t::observation(ukf_t::i_x, ukf_t::i_r, ukf_t::i_z);
        // Capture output into Z.
        ukf_t::Z.col(1 + 2*ukf_t::n_x + j) = ukf_t::i_z;
        
        // mean MINUS y*sqrt(R)
        // Populate interface vectors.
        ukf_t::i_x = ukf_t::X.col(0);
        ukf_t::i_r = -ukf_t::Xr.col(j);
        ukf_t::i_z.setZero(ukf_t::n_z);
        // Run observation function.
        ukf_t::observation(ukf_t::i_x, ukf_t::i_r, ukf_t::i_z);
        // Capture output into Z.
        ukf_t::Z.col(1 + 2*ukf_t::n_x + ukf_t::n_z + j) = ukf_t::i_z;
    }



    // Calculate predicted state mean and covariance.
    
    // Predicted state mean is a weighted average: sum(wm.*X) over all sigma points.
    // Can be calculated via matrix multiplication with wm vector.
    ukf_t::x.noalias() = ukf_t::X * ukf_t::wm_X;

    // Predicted state covariance is a weighted average: sum(wc.*(X-x)(X-x)') over all sigma points.
    // This can be calculated via matrix multiplication (X-x)wc(X-x)'
    ukf_t::X.colwise() -= ukf_t::x;

    ukf_t::P.setZero();
    for(uint32_t j = 0; j < ukf_t::n_X; ++j)
    {
        // Calculate X-x at this sigma column.
        ukf_t::t_x = ukf_t::X.col(j) - ukf_t::x;
        // Calculate outer product.
        ukf_t::t_xx.noalias() = ukf_t::t_x * ukf_t::t_x.transpose();
        // Apply weight to outer product.
        ukf_t::t_xx *= ukf_t::wc(j);
        // Sum into predicted state covariance.
        ukf_t::P += ukf_t::t_xx;
    } 
}
void ukf_t::predict()
{
    

    
}
void ukf_t::update(observer_id_t observer_id, const Eigen::VectorXd& z)
{





    // Calculate mean/covariance for predicted observation, as well as cross covariance for predicted state and predicted observation.
    
    // Predicted observation mean is a weighted average: sum(wm.*Z) over all sigma points.
    // Can be calculated via matrix multiplication with wm vector.
    observer.z.noalias() = observer.Z * observer.wm;

    // Predicted observation covariance is a weighted average: sum(wc.*(Z-z)(Z-z)') over all sigma points.
    // Predicted state/observation cross-covariance is a weighted average: sum(wc.*(X-x)(Z-z)') over all sigma points.
    observer.S.setZero();
    observer.C.setZero();
    for(uint32_t j = 0; j < observer.n_Z; ++j)
    {
        // Calculate X-x at this sigma column.
        // X is broken down storage-wize into x, +/- Xp, +/-Xq.
        if(j == 0)
        {
            ukf_t::t_x.setZero();
        }
        else if(j < (ukf_t::n_x + 1))
        {
            ukf_t::t_x = ukf_t::Xp.col(j-1);
        }
        else if(j < (2*ukf_t::n_x + 1))
        {
            ukf_t::t_x = -ukf_t::Xp.col(j-(1+ukf_t::n_x));
        }
        else
        {
            ukf_t::t_x.setZero();
        }
        // Calculate Z-z at this sigma column.
        observer.t_z = observer.Z.col(j) - observer.z;
        // Calculate outer products.
        observer.t_zz.noalias() = observer.t_z * observer.t_z.transpose();
        observer.t_xz.noalias() = ukf_t::t_x * observer.t_z.transpose();
        // Apply weight to outer products.
        observer.t_zz *= observer.wc(j);
        observer.t_xz *= observer.wc(j);
        // Sum into predicted observation covariance.
        observer.S += observer.t_zz;
        // Sum into cross covariance.
        observer.C += observer.t_xz;
    }

    // Calculate Kalman gain.
    observer.t_zz = observer.S.inverse();
    observer.K.noalias() = observer.C * observer.t_zz;

    // Update state.
    observer.t_z = z - observer.z;
    ukf_t::x.noalias() += observer.K * observer.t_z;

    // Update covariance.
    observer.t_xz.noalias() = observer.K * observer.S;
    ukf_t::P.noalias() -= observer.t_xz * observer.K.transpose();
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