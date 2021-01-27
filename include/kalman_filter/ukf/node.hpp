#ifndef KALMAN_FILTER___UKF___NODE_H
#define KALMAN_FILTER___UKF___NODE_H

#include <kalman_filter/ukf/model_plugin.hpp>

#include <ros/ros.h>

namespace kalman_filter {
namespace ukf {

class node_t
{
public:
    node_t();

    void run();

private:
    std::unique_ptr<ros::NodeHandle> m_node;
};

}}

#endif