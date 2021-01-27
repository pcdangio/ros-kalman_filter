#include <kalman_filter/ukf/node.hpp>

using namespace kalman_filter::ukf;

node_t::node_t()
{
    
}

void node_t::run()
{
    auto plugin = kalman_filter::model_plugin_t::load("/home/pcdangio/projects/ros/devel/lib/libtest_kf.so");
}