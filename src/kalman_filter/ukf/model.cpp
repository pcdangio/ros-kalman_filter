#include <kalman_filter/ukf/model.hpp>

using namespace kalman_filter::ukf;

model_t::model_t(uint32_t dimensions)
    : dimensions(dimensions)
{
    // Initialize Q.
    model_t::Q.setIdentity(dimensions, dimensions);
}