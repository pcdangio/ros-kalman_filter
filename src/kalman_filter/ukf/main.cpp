//#include <kalman_filter/ukf/node.hpp>

#include <eigen3/Eigen/Dense>
#include <iostream>

int32_t main(int32_t argc, char** argv)
{
    Eigen::MatrixXd m;
    m.setZero(4,4);
    m(0,0) = 2;
    m(0,1) = 1;
    m(1,0) = 1;
    m(1,1) = 2;
    m(2,2) = 3;
    m(2,3) = 2;
    m(3,2) = 2;
    m(3,3) = 3;

    std::cout << m << std::endl;

    Eigen::LLT<Eigen::MatrixXd> m_llt;
    m_llt.compute(m);

    Eigen::MatrixXd lower = m_llt.matrixL();

    std::cout << lower << std::endl;

    // // Initialize ROS.
    // ros::init(argc, argv, "ukf");

    // // Create UKF node.
    // kalman_filter::ukf::node_t node;

    // // Run node.
    // node.run();

    return 0;
}