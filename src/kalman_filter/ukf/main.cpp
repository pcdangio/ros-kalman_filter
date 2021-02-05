#include <kalman_filter/ukf/ukf.hpp>

#include <eigen3/Eigen/Dense>
#include <iostream>

void transition_function(const Eigen::VectorXd& xp, const Eigen::VectorXd& q, Eigen::VectorXd& x)
{

}
void h0(const Eigen::VectorXd& x, const Eigen::VectorXd& r, Eigen::VectorXd& z)
{

}

int32_t main(int32_t argc, char** argv)
{
    kalman_filter::ukf::ukf_t ukf(3, transition_function);

    ukf.add_observer(0, 2, h0);
    ukf.remove_observer(0);

    Eigen::VectorXd x_o;
    x_o.setZero(3);
    Eigen::MatrixXd P_o;
    P_o.setIdentity(3,3);
    ukf.initialize(x_o, P_o);

    // // Initialize ROS.
    // ros::init(argc, argv, "ukf");

    // // Create UKF node.
    // kalman_filter::ukf::node_t node;

    // // Run node.
    // node.run();

    return 0;
}