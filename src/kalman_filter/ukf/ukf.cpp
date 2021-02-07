#include <kalman_filter/ukf/ukf.hpp>

using namespace kalman_filter::ukf;

// CONSTRUCTORS
ukf_t::ukf_t(uint32_t dimensions, function_t prediction_function)
{
    // Store prediction function.
    ukf_t::f = prediction_function;

    // Store dimension sizes.
    ukf_t::n_x = dimensions;
    ukf_t::n_xa = 2*ukf_t::n_x;
    ukf_t::n_X = 1 + 2*ukf_t::n_xa;

    // Allocate weight vectors.
    ukf_t::wm.setZero(ukf_t::n_X);
    ukf_t::wc.setZero(ukf_t::n_X);

    // Allocate variable components.
    ukf_t::x.setZero(ukf_t::n_x);
    ukf_t::P.setIdentity(ukf_t::n_x, ukf_t::n_x);
    ukf_t::Q.setIdentity(ukf_t::n_x, ukf_t::n_x);
    ukf_t::Xp.setZero(ukf_t::n_x, ukf_t::n_x);
    ukf_t::Xq.setZero(ukf_t::n_x, ukf_t::n_x);
    ukf_t::X.setZero(ukf_t::n_x, ukf_t::n_X);

    // Allocate interface components.
    ukf_t::i_xp.setZero(ukf_t::n_x);
    ukf_t::i_x.setZero(ukf_t::n_x);
    ukf_t::i_q.setZero(ukf_t::n_x);

    // Allocate temporaries.
    ukf_t::t_x.setZero(ukf_t::n_x);
    ukf_t::t_xx.setZero(ukf_t::n_x, ukf_t::n_x);

    // Set default parameters.
    ukf_t::alpha = 0.001;
    ukf_t::kappa = 3 - ukf_t::n_x;
    ukf_t::beta = 2;
}

// OBSERVER MANAGEMENT
void ukf_t::add_observer(observer_id_t id, uint32_t dimensions, function_t observation_function)
{
    // Create a new observer and grab a reference to it (or grab existing observer to replace it)
    observer_t& observer = ukf_t::observers[id];

    // Store observer's function.
    observer.h = observation_function;

    // Store dimension sizes.
    observer.n_z = dimensions;
    observer.n_za = ukf_t::n_x + observer.n_z;
    observer.n_Z = 1 + 2*observer.n_za;

    // Allocate weight vectors.
    observer.wm.setZero(observer.n_Z);
    observer.wc.setZero(observer.n_Z);

    // Allocate observer components.
    observer.R.setIdentity(observer.n_z, observer.n_z);
    observer.Xr.setZero(observer.n_z, observer.n_z);
    observer.Z.setZero(observer.n_z, observer.n_Z);
    observer.z.setZero(observer.n_z);
    observer.S.setZero(observer.n_z, observer.n_z);
    observer.C.setZero(ukf_t::n_x, observer.n_z);
    observer.K.setZero(ukf_t::n_x, observer.n_z);

    // Allocate interface components.
    observer.i_r.setZero(observer.n_z);
    observer.i_z.setZero(observer.n_z);

    // Allocate temporaries.
    observer.t_z.setZero(observer.n_z);
    observer.t_zz.setZero(observer.n_z, observer.n_z);
    observer.t_xz.setZero(ukf_t::n_x, observer.n_z);
}
void ukf_t::remove_observer(observer_id_t id)
{
    // Remove observer.
    ukf_t::observers.erase(id);
}

// FILTER METHODS
void ukf_t::initialize(const Eigen::VectorXd& initial_state, const Eigen::MatrixXd& initial_covariance)
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
void ukf_t::predict()
{
    // Calculate lambda for this iteration (user can change parameters between iterations)
    double lambda = ukf_t::alpha * ukf_t::alpha * (static_cast<double>(ukf_t::n_xa) + ukf_t::kappa) - static_cast<double>(ukf_t::n_xa);

    // Calculate weight vectors for mean and covariance averaging.
    // Set mean recovery weight vector.
    ukf_t::wm.fill(1.0 / (2.0 * (static_cast<double>(ukf_t::n_xa) + lambda)));
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
    // y*sqrt(R) not set in predict as not used.

    // Populate Xp and Xq input sigma matrices.
    
    // Calculate square root of P using Cholseky Decomposition
    ukf_t::llt.compute(ukf_t::P);
    // Fill +sqrt(P) block of Xp.
    ukf_t::Xp = ukf_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukf_t::Xp *= std::sqrt(static_cast<double>(ukf_t::n_xa) + lambda);

    // Calculate square root of Q using Cholseky Decomposition.
    ukf_t::llt.compute(ukf_t::Q);
    // Fill +sqrt(Q) block of Xq.
    ukf_t::Xq = ukf_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukf_t::Xq *= std::sqrt(static_cast<double>(ukf_t::n_xa) + lambda);

    // Evaluate sigma points through transition function.

    // Pass first set of sigma points, which is just the mean.
    // Populate interface vectors.
    ukf_t::i_xp = ukf_t::x;
    ukf_t::i_q.setZero(ukf_t::n_x);
    ukf_t::i_x.setZero(ukf_t::n_x);
    // Run transition function.
    ukf_t::f(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
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
        ukf_t::f(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.col(1 + j) = ukf_t::i_x;
        
        // mean MINUS y*sqrt(P)
        // Populate interface vectors.
        ukf_t::i_xp = ukf_t::x - ukf_t::Xp.col(j);
        ukf_t::i_q.setZero(ukf_t::n_x);
        ukf_t::i_x.setZero(ukf_t::n_x);
        // Run transition function.
        ukf_t::f(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
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
        ukf_t::f(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.col(1 + 2*ukf_t::n_x + j) = ukf_t::i_x;
        
        // mean MINUS y*sqrt(Q)
        // Populate interface vectors.
        ukf_t::i_xp = ukf_t::x;
        ukf_t::i_q = -ukf_t::Xq.col(j);
        ukf_t::i_x.setZero(ukf_t::n_x);
        // Run transition function.
        ukf_t::f(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.col(1 + 3*ukf_t::n_x + j) = ukf_t::i_x;
    }

    // Calculate predicted state mean and covariance.
    
    // Predicted state mean is a weighted average: sum(wm.*X) over all sigma points.
    // Can be calculated via matrix multiplication with wm vector.
    ukf_t::x.noalias() = ukf_t::X * ukf_t::wm;

    // Predicted state covariance is a weighted average: sum(wc.*(X-x)(X-x)') over all sigma points.
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
void ukf_t::update(observer_id_t observer_id, const Eigen::VectorXd& z)
{
    // Get the observer by ID.
    auto entry = ukf_t::observers.find(observer_id);
    // Check if the observer was found.
    if(entry == ukf_t::observers.end())
    {
        throw std::runtime_error("observer with specified id does not exist");
    }
    // Store reference to observer.
    observer_t& observer = entry->second;

    // Verify that observation vector size matches.
    if(z.size() != observer.n_z)
    {
        throw std::runtime_error("observation vector has incorrect size");
    }

    // NOTE: Observer needs it's own lambda and weights as it's Z sigma matrix has a unique size and is different from X sigma matrix.

    // Calculate lambda for this iteration (user can change parameters between iterations)
    double lambda = ukf_t::alpha * ukf_t::alpha * (static_cast<double>(observer.n_za) + ukf_t::kappa) - static_cast<double>(observer.n_za);

    // Calculate weight vectors for mean and covariance averaging.
    // Set mean recovery weight vector.
    observer.wm.fill(1.0 / (2.0 * (static_cast<double>(observer.n_za) + lambda)));
    observer.wm(0) *= 2.0 * lambda;
    // Copy wc from wm and update first element.
    observer.wc = observer.wm;
    observer.wc(0) += (1.0 - ukf_t::alpha*ukf_t::alpha + ukf_t::beta);

    // Set up input sigma matrices.
    // NOTE: This implementation segments out the input sigma matrix for efficiency:
    // [u u+y*sqrt(P) u-y*sqrt(P) 0           0           0           0          ]
    // [0 0           0           u+y(sqrt(Q) u-y*sqrt(Q) 0           0          ]
    // [0 0           0           0           0           u+y*sqrt(R) u-y*sqrt(R)]
    // u is stored in x
    // y*sqrt(P) stored in Xp
    // y*sqrt(Q) not set in update as not used.
    // y*sqrt(R) stored in Xr

    // Populate Xp and Xr input sigma matrices.
 
    // Calculate square root of P using Cholseky Decomposition.
    ukf_t::llt.compute(ukf_t::P);
    // Fill +sqrt(P) block of Xp.
    ukf_t::Xp = ukf_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukf_t::Xp *= std::sqrt(static_cast<double>(observer.n_za) + lambda);

    // Calculate square root of R using Cholseky Decomposition.
    ukf_t::llt.compute(observer.R);
    // Fill +sqrt(R) block of Xr.
    observer.Xr = ukf_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    observer.Xr *= std::sqrt(static_cast<double>(observer.n_za) + lambda);

    // Evaluate sigma points through observation function.

    // Pass first set of sigma points, which is just the mean.
    // Populate interface vectors.
    ukf_t::i_x = ukf_t::x;
    observer.i_r.setZero(observer.n_z);
    observer.i_z.setZero(observer.n_z);
    // Run observation function.
    observer.h(ukf_t::i_x, observer.i_r, observer.i_z);
    // Capture output into Z.
    observer.Z.col(0) = observer.i_z;

    // Pass second set of sigma points, which focuses on covariance P.
    for(uint32_t j = 0; j < ukf_t::Xp.cols(); ++j)
    {
        // mean PLUS y*sqrt(P)
        // Populate interface vectors.
        ukf_t::i_x = ukf_t::x + ukf_t::Xp.col(j);
        observer.i_r.setZero(observer.n_z);
        observer.i_z.setZero(observer.n_z);
        // Run observation function.
        observer.h(ukf_t::i_x, observer.i_r, observer.i_z);
        // Capture output into Z.
        observer.Z.col(1 + j) = observer.i_z;
        
        // mean MINUS y*sqrt(P)
        // Populate interface vectors.
        ukf_t::i_x = ukf_t::x - ukf_t::Xp.col(j);
        observer.i_r.setZero(observer.n_z);
        observer.i_z.setZero(observer.n_z);
        // Run observation function.
        observer.h(ukf_t::i_x, observer.i_r, observer.i_z);
        // Capture output into Z.
        observer.Z.col(1 + ukf_t::n_x + j) = observer.i_z;
    }

    // Pass third set of sigma points, which focuses on covariance R.
    for(uint32_t j = 0; j < observer.Xr.cols(); ++j)
    {
        // mean PLUS y*sqrt(R)
        // Populate interface vectors.
        ukf_t::i_x = ukf_t::x;
        observer.i_r = observer.Xr.col(j);
        observer.i_z.setZero(observer.n_z);
        // Run observation function.
        observer.h(ukf_t::i_x, observer.i_r, observer.i_z);
        // Capture output into Z.
        observer.Z.col(1 + 2*ukf_t::n_x + j) = observer.i_z;
        
        // mean MINUS y*sqrt(Q)
        // Populate interface vectors.
        ukf_t::i_x = ukf_t::x;
        observer.i_r = -observer.Xr.col(j);
        observer.i_z.setZero(observer.n_z);
        // Run observation function.
        observer.h(ukf_t::i_x, observer.i_r, observer.i_z);
        // Capture output into Z.
        observer.Z.col(1 + 2*ukf_t::n_x + observer.n_z + j) = observer.i_z;
    }

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
        // X is broken down storage-wize into x, +/- Xp, +/-Xr.
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

// ACCESS
Eigen::MatrixXd& ukf_t::R(observer_id_t observer_id)
{
    // Get the observer by ID.
    auto observer_entry = ukf_t::observers.find(observer_id);

    // Check if the observer was found.
    if(observer_entry == ukf_t::observers.end())
    {
        throw std::runtime_error("observer with specified id does not exist");
    }

    // Return reference to observer's R.
    return observer_entry->second.R;
}
const Eigen::VectorXd& ukf_t::state() const
{
    return ukf_t::x;
}
const Eigen::MatrixXd& ukf_t::covariance() const
{
    return ukf_t::P;
}