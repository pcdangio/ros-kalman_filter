#ifndef KALMAN_FILTER___MODEL_PLUGIN_H
#define KALMAN_FILTER___MODEL_PLUGIN_H

#include <eigen3/Eigen/Dense>

#include <memory>

namespace kalman_filter {

class model_plugin_t
{
public:
    uint32_t n_state_variables() const;
    uint32_t n_measurement_variables() const;

    virtual void initialize_state(Eigen::VectorXd& state);

    const Eigen::MatrixXd& q() const;
    const Eigen::MatrixXd& r() const;
    const Eigen::VectorXd& z() const;
    const Eigen::MatrixXd& m() const;

    void clear_measurements();

protected:
    model_plugin_t(uint32_t n_state_variables, uint32_t n_measurement_variables);
    static std::shared_ptr<model_plugin_t> load_base(const std::string& path);

    Eigen::MatrixXd m_q;
    Eigen::MatrixXd m_r;

    void new_measurement(uint32_t index, double value);

private:
    uint32_t m_n_state_variables;
    uint32_t m_n_measurement_variables;

    Eigen::VectorXd m_z;
    Eigen::MatrixXd m_m;
};

#define REGISTER_MODEL_PLUGIN(class_name) extern "C" class_name* instantiate() {return new class_name();}

}

#endif