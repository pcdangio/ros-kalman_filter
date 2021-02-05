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
    ukf_t::wm(0) = lambda / (static_cast<double>(ukf_t::n_xa) + lambda);
    ukf_t::wm.tail(ukf_t::n_X - 1).fill(1.0 / (2.0 * (static_cast<double>(ukf_t::n_xa) + lambda)));
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

    // Set up Xp input sigma matrix.
    // Calculate square root of P using Cholseky Decomposition.
    ukf_t::llt.compute(ukf_t::P);
    // Fill +sqrt(P) block of Xp.
    ukf_t::Xp = ukf_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukf_t::Xp *= std::sqrt(static_cast<double>(ukf_t::n_xa) + lambda);

    // Set up Xq input sigma matrix.
    // Calculate square root of Q using Cholseky Decomposition.
    ukf_t::llt.compute(ukf_t::Q);
    // Fill +sqrt(Q) block of Xq.
    ukf_t::Xq = ukf_t::llt.matrixL();
    // Apply sqrt(n+lambda) to entire matrix.
    ukf_t::Xq *= std::sqrt(static_cast<double>(ukf_t::n_xa) + lambda);

    // Pass input sigma points through transition function to populate X sigma matrix.

    // Pass first set of sigma points, which is just the mean.
    // Populate interface vectors.
    ukf_t::i_xp = ukf_t::x;
    ukf_t::i_q.setZero();
    ukf_t::i_x.setZero();
    // Run transition function.
    ukf_t::f(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
    // Capture output into X.
    ukf_t::X.block(0, 0, ukf_t::n_x, 1) = ukf_t::i_x;

    // Pass second set of sigma points, which focuses on covariance P.
    for(uint32_t j = 0; j < ukf_t::Xp.cols(); ++j)
    {
        // mean PLUS y*sqrt(P)
        // Populate interface vectors.
        ukf_t::i_xp = ukf_t::x + ukf_t::Xp.col(j);
        ukf_t::i_q.setZero();
        ukf_t::i_x.setZero();
        // Run transition function.
        ukf_t::f(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.block(0, 1+j, ukf_t::n_x, 1) = i_x;
        
        // mean MINUS y*sqrt(P)
        // Populate interface vectors.
        ukf_t::i_xp = ukf_t::x - ukf_t::Xp.col(j);
        ukf_t::i_q.setZero();
        ukf_t::i_x.setZero();
        // Run transition function.
        ukf_t::f(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.block(0, 1+j+ukf_t::n_x, ukf_t::n_x, 1) = i_x;
    }

    // Pass third set of sigma points, which focuses on covariance Q.
    for(uint32_t j = 0; j < ukf_t::Xq.cols(); ++j)
    {
        // mean PLUS y*sqrt(Q)
        // Populate interface vectors.
        ukf_t::i_xp = ukf_t::x;
        ukf_t::i_q = ukf_t::Xq.col(j);
        ukf_t::i_x.setZero();
        // Run transition function.
        ukf_t::f(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.block(0, 1+2*ukf_t::n_x+j, ukf_t::n_x, 1) = i_x;
        
        // mean MINUS y*sqrt(Q)
        // Populate interface vectors.
        ukf_t::i_xp = ukf_t::x;
        ukf_t::i_q = -ukf_t::Xq.col(j);
        ukf_t::i_x.setZero();
        // Run transition function.
        ukf_t::f(ukf_t::i_xp, ukf_t::i_q, ukf_t::i_x);
        // Capture output into X.
        ukf_t::X.block(0, 1+2*ukf_t::n_x+j+ukf_t::n_x, ukf_t::n_x, 1) = i_x;
    }

    // Calculate predicted state and covariance.
    
    // Predicted state is a weighted average: sum(wm.*X) over all sigma points.
    // Can be calculated via matrix multiplication with wm vector.
    ukf_t::x.noalias() = ukf_t::X * ukf_t::wm;

    // Predicted covariance is a weighted average: sum(wc.*(X-x)(X-x)') over all sigma points.
    ukf_t::P.setZero();
    for(uint32_t j = 0; j < ukf_t::n_X; ++j)
    {
        // Calculate X-x at this sigma column.
        ukf_t::t_x = ukf_t::X.col(j) - ukf_t::x;
        // Calculate outer product.
        ukf_t::t_xx.noalias() = ukf_t::t_x * ukf_t::t_x.transpose();
        // Apply weight to outer product.
        ukf_t::t_xx *= ukf_t::wc(j);
        // Sum into predicted covariance.
        ukf_t::P += ukf_t::t_xx;
    }
}
void ukf_t::update(observer_id_t observer, const Eigen::VectorXd& z)
{

}

// ACCESS
Eigen::MatrixXd& ukf_t::R(observer_id_t id)
{
    // Get the observer by ID.
    auto observer_entry = ukf_t::observers.find(id);

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