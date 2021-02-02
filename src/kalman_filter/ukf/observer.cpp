#include <kalman_filter/ukf/observer.hpp>

using namespace kalman_filter::ukf;

// CONSTRUCTORS
observer_t::observer_t(uint32_t id, uint32_t dimensions)
    : id(id), dimensions(dimensions)
{
    // Initialize covariance matrix to identity (balanced with default Q).
    observer_t::covariance.setIdentity(dimensions, dimensions);
}