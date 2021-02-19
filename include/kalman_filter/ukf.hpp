#ifndef KALMAN_FILTER___UKF_H
#define KALMAN_FILTER___UKF_H

#include <kalman_filter/base.hpp>

namespace kalman_filter {

class ukf_t
    :public base_t
{
public:
    ukf_t(uint32_t n_variables, uint32_t n_observers);

    // MODEL FUNCTIONS
    /// \brief Predicts a new state by transitioning from a prior state.
    /// \param xp The prior state to transition from.
    /// \param x (OUTPUT) The predicted new state.
    /// \note This function must not make changes to any external object.
    virtual void state_transition(const Eigen::VectorXd& xp, Eigen::VectorXd& x) const = 0;
    /// \brief Predicts an observation from a state.
    /// \param x The state to predict an observation from.
    /// \param z (OUTPUT) The predicted observation.
    /// \note This function must not make changes to any external object.
    virtual void observation(const Eigen::VectorXd& x, Eigen::VectorXd& z) const = 0;

    // FILTER METHODS
    void iterate() override;

    // PARAMETERS
    /// \brief The alpha parameter of the UKF.
    double_t alpha;
    /// \brief The kappa parameter of the UKF.
    double_t kappa;
    /// \brief The beta parameter of the UKF.
    double_t beta;

private:
    // DIMENSIONS
    /// \brief The number of sigma points.
    uint32_t n_s;

    // STORAGE: WEIGHTS
    /// \brief The mean recovery weight vector.
    Eigen::VectorXd wm;
    /// \brief The covariance recovery weight vector.
    Eigen::VectorXd wc;

    // STORAGE: SIGMA
    /// \brief The evaluated variable sigma matrix.
    Eigen::MatrixXd X;
    /// \brief The evaluated observation sigma matrix.
    Eigen::MatrixXd Z;

    // STORAGE: INTERFACES
    /// \brief An interface to the prior state vector.
    Eigen::VectorXd i_xp;
    /// \brief An interface to the current state vector.
    Eigen::VectorXd i_x;
    /// \brief An interface to the predicted observation vector.
    Eigen::VectorXd i_z;

    // STORAGE: TEMPORARIES
    /// \brief A temporary working matrix of size z,z.
    Eigen::MatrixXd t_zz;
    /// \brief A temporary working matrix of size x,s.
    Eigen::MatrixXd t_xs;
    /// \brief A temporary working matrix of size z,s.
    Eigen::MatrixXd t_zs;

    // UTILITY
    /// \brief An LLT object for storing results of Cholesky decompositions.
    mutable Eigen::LLT<Eigen::MatrixXd> llt;
};

}

#endif