#ifndef KALMAN_FILTER___MODEL_PLUGIN_H
#define KALMAN_FILTER___MODEL_PLUGIN_H

#include <eigen3/Eigen/Dense>

namespace kalman_filter {

class model_plugin_t
{
public:
    model_plugin_t(uint32_t n_state_variables, uint32_t n_measurement_variables);

    uint32_t n_state_variables() const;
    uint32_t n_measurement_variables() const;

    const Eigen::MatrixXd& q() const;
    const Eigen::MatrixXd& r() const;
    const Eigen::VectorXd& z() const;
    const Eigen::MatrixXd& m() const;

    void clear_measurements();

protected:
    Eigen::MatrixXd m_q;
    Eigen::MatrixXd m_r;

    void new_measurement(uint32_t index, double value);

private:
    uint32_t m_n_state_variables;
    uint32_t m_n_measurement_variables;

    Eigen::VectorXd m_z;
    Eigen::MatrixXd m_m;
};

}

#endif