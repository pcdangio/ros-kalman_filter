#include <kalman_filter/kf.hpp>

using namespace kalman_filter;

// CONSTRUCTORS
kf_t::kf_t(uint32_t n_variables, uint32_t n_inputs, uint32_t n_observers)
    : base_t(n_variables, n_observers)
{
    // Store dimensions.
    kf_t::n_u = n_inputs;

    // Initialize model matrices.
    kf_t::A.setIdentity(kf_t::n_x, kf_t::n_x);
    kf_t::B.setZero(kf_t::n_x, kf_t::n_u);
    kf_t::H.setZero(kf_t::n_z, kf_t::n_x);

    // Initialize input vector.
    kf_t::u.setZero(kf_t::n_u);

    // Initialize temporaries.
    kf_t::t_x.setZero(kf_t::n_x);
    kf_t::t_xx.setZero(kf_t::n_x, kf_t::n_x);
    kf_t::t_zx.setZero(kf_t::n_z, kf_t::n_x);
}

// FILTER METHODS
void kf_t::iterate()
{
    // ---------- STEP 1: PREDICT ----------

    // Predict state.
    kf_t::t_x.noalias() = kf_t::A * kf_t::x;
    kf_t::x.noalias() = kf_t::B * kf_t::u;
    kf_t::x += kf_t::t_x;

    // Predict covariance.
    kf_t::t_xx.noalias() = kf_t::A * kf_t::P;
    kf_t::P.noalias() = kf_t::t_xx * kf_t::A.transpose();
    kf_t::P += kf_t::Q;

    // ---------- STEP 2: UPDATE ----------

    // Check if update is necessary.
    if(kf_t::has_observations())
    {
        // Calculate predicted observation covariance.
        kf_t::t_zx.noalias() = kf_t::H * kf_t::P;
        kf_t::S.noalias() = kf_t::t_zx * kf_t::H.transpose();
        kf_t::S += kf_t::R;

        // Calculate predicted state/observation cross covariance.
        kf_t::C.noalias() = kf_t::P * kf_t::H.transpose();

        // Perform masked kalman update.
        kf_t::masked_kalman_update();
    }
}
void kf_t::new_input(uint32_t input_index, double_t input)
{
    // Verify index exists.
    if(!(input_index < kf_t::n_u))
    {
        throw std::runtime_error("failed to set new input (input_index out of range)");
    }

    // Store input.
    kf_t::u(input_index) = input;
}

// ACCESS
uint32_t kf_t::n_inputs() const
{
    return kf_t::n_u;
}