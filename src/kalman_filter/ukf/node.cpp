#include <kalman_filter/ukf/node.hpp>

using namespace kalman_filter::ukf;

node_t::node_t()
{
    
}

void node_t::run()
{
    auto plugin = kalman_filter::ukf::model_plugin_t::load_ukf_model("/home/pcdangio/projects/ros/devel/lib/libtest_kf.so");

    Eigen::VectorXd hah;
    hah.setRandom(5);
    plugin->state_transition(hah, hah);
}