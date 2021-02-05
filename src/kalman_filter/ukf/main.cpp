#include <kalman_filter/ukf/ukf.hpp>

#include <eigen3/Eigen/Dense>
#include <iostream>

void transition_function(const Eigen::VectorXd& xp, const Eigen::VectorXd& q, Eigen::VectorXd& x)
{
    x(0) = xp(0) + 1.0 + q(0);
    x(1) = xp(1) + 2.0 + q(1);
    x(2) = xp(2) + 3.0 + q(2);
}
void h0(const Eigen::VectorXd& x, const Eigen::VectorXd& r, Eigen::VectorXd& z)
{
    z(0) = x(0) + r(0);
    z(1) = x(1) + r(1);
}

int32_t main(int32_t argc, char** argv)
{
    kalman_filter::ukf::ukf_t ukf(3, transition_function);

    ukf.Q.setIdentity();

    ukf.add_observer(0, 2, h0);
    ukf.R(0).setIdentity();

    Eigen::VectorXd x_o;
    x_o.setZero(3);
    Eigen::MatrixXd P_o;
    P_o.setIdentity(3,3);
    ukf.initialize(x_o, P_o);

    ukf.predict();

    std::cout << ukf.state() << std::endl << std::endl;
    std::cout << ukf.covariance() << std::endl;

    Eigen::Vector2d z;
    ukf.update(0, z);

    Eigen::MatrixXd L;
    L.setZero(10, 10);

    Eigen::Block<Eigen::MatrixXd> l = L.block(0, 0, 3, 3);
    l.fill(1);

    L(1,1) = 2;

    std::cout << L << std::endl;

    // // Initialize ROS.
    // ros::init(argc, argv, "ukf");

    // // Create UKF node.
    // kalman_filter::ukf::node_t node;

    // // Run node.
    // node.run();

    return 0;
}