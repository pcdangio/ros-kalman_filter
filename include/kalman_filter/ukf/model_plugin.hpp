#ifndef KALMAN_FILTER___UKF___MODEL_PLUGIN_H
#define KALMAN_FILTER___UKF___MODEL_PLUGIN_H

#include <kalman_filter/model_plugin.hpp>

namespace kalman_filter {
namespace ukf {

class model_plugin_t
    : public kalman_filter::model_plugin_t
{
public:
    model_plugin_t(uint32_t n_state_variables, uint32_t n_measurement_variables);
    
    virtual void state_transition(const Eigen::VectorXd& xp, Eigen::VectorXd& x) const = 0;
    virtual void predict_measurement(const Eigen::VectorXd& x, Eigen::VectorXd& z) const = 0;
};

#define REGISTER_MODEL_PLUGIN(class_name) extern "C" kalman_filter::ukf::model_plugin_t* create_model_plugin() {return new class_name();}

}}

#endif