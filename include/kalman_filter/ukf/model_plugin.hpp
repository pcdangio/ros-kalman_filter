#ifndef KALMAN_FILTER___UKF___MODEL_PLUGIN_H
#define KALMAN_FILTER___UKF___MODEL_PLUGIN_H

#include <kalman_filter/model_plugin.hpp>

namespace kalman_filter {
namespace ukf {

class model_plugin_t
    : public kalman_filter::model_plugin_t
{
public:
    static std::shared_ptr<ukf::model_plugin_t> load_ukf_model(const std::string& plugin_path);

    virtual void state_transition(const Eigen::VectorXd& xp, Eigen::VectorXd& x) const = 0;
    virtual void predict_measurement(const Eigen::VectorXd& x, Eigen::VectorXd& z) const = 0;

protected:
    model_plugin_t(uint32_t n_state_variables, uint32_t n_measurement_variables);
};

}}

#endif