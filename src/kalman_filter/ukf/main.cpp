#include <kalman_filter/ukf/node.hpp>

int32_t main(int32_t argc, char** argv)
{
    // Initialize ROS.
    ros::init(argc, argv, "ukf");

    // Create UKF node.
    kalman_filter::ukf::node_t node;

    // Run node.
    node.run();

    return 0;
}