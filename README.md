# kalman_filter

## Overview

The kalman_filter ROS package provides C++ libraries for several types of Kalman Filters that can be used for state estimation:

1. **Kalman Filter (KF):** for linear systems with additive noise
2. **Unscented Kalman Filter (UKF):** for nonlinear systems with additive noise
3. **Unscented Kalman Filter - Augmented (UKFA):** for nonlinear systems with non-additive noise

The libraries require minimal effort from the user to implement. The only steps the user must take to use the filters are:

- Provide the state transition and observation models
- Set the process covariance (Q) and observation covariance (R)
- Pass observations into the filter

The filter will internally handle all other calculations/algorithms.

**Key features of this library include:**

- Extremely easy to implement and use
- Very high memory/computation efficiency
- Gracefully handles observers with different data rates (e.g. 5Hz GPS and 200Hz IMU)

See the [kalman_filter_examples](https://github.com/pcdangio/ros-kalman_filter_examples) package for how to use the various filters in the package.

## Table of Contents

- [Installation](#1-installation): Instructions for installing the package from source.
- [Usage](#2-usage): Instructions for using the various filters, including common tips.
  - [Kalman Filter](#21-kalman-filter-kf)
  - [Unscented Kalman Filter](#22-unscented-kalman-filter-ukf)
  - [Unscented Kalman Filter - Augmented](#23-unscented-kalman-filter---augmented-ukfa)

## 1: Installation

**Dependencies:**

- ROS Melodic or higher
- Eigen 3 (comes with ROS)

**Download/Build:**

You may use the following commands to clone and build the package:

```bash
# Switch to your catkin workspace source folder.
cd catkin_workspace/src

# Clone package into source folder.
git clone https://github.com/pcdangio/ros-kalman_filter.git kalman_filter

# Switch back to catkin workspace root folder.
cd ..

# Build with catkin_make.
# NOTE: Using "-DCMAKE_BUILD_TYPE=Release" significantly improves performance of the libraries.
catkin_make --pkg=kalman_filter -DCMAKE_BUILD_TYPE=Release
```

## 2: Usage

The following sections outline the usage for each filter included in this package. Some general notes for usage that are common to all filter types include:

- It is strongly recommended you have a basic understanding of Kalman Filtering before using this package.
- Make sure to include `kalman_filter` as a dependency in your project's `CMakeLists.txt`. See [kalman_filter_examples](https://github.com/pcdangio/ros-kalman_filter_examples/blob/main/CMakeLists.txt) for an example `CMakeLists.txt`.
- For best performance, you should call `iterate()` to run the filter at least as fast as your fastest observer. For example, if you have a 5Hz GPS and a 200Hz IMU, you should run `iterate()` at a minimum of 200Hz.
- You may add a new observation to the filter at any time using the `new_observation(observer_index,value)` method. This approach provides two primary advantages:
  - Observations can be provided to the filter at variable/different rates
  - The filter only performs update calculations on available observations, maximizing efficiency

### 2.1: Kalman Filter (KF)

The standard Kalman Filter can be used for state estimation of linear systems with additive noise.

The following code snippet demonstrates a very minimal example of how to use the KF library. More KF-specific examples can be found under [kalman_filter_examples](https://github.com/pcdangio/ros-kalman_filter_examples/tree/main/src/kf).

```cpp
#include <kalman_filter/kf.hpp>

int32_t main(int32_t argc, char** argv)
{
    // Set up a new KF that has a single state, a single input, and a single observer.
    kalman_filter::kf_t kf(1,1,1);

    // Populate the model matrices accordingly.
    kf.A(0,0) = 1.0;    // State Transition
    kf.B(0,0) = 1.0;    // Control Input
    kf.H(0,0) = 1.0;    // Observation

    // Set up process and observation covariance matrices.
    kf.Q(0,0) = 0.01;   // Process Covariance
    kf.R(0,0) = 1;      // Observation Covariance

    // OPTIONAL: Set the initial state and covariance of the model.
    Eigen::VectorXd x_o(1);
    x_o(0) = 2;
    Eigen::MatrixXd P_o(1,1);
    P_o(0,0) = 0.5;
    kf.initialize_state(x_o, P_o);

    // The following code can be run continuously in a loop:

    // Calculate some new input:
    double_t u = 5.0;
    // Pass input into the filter.
    // The index specifies the position of the input in the control input vector.
    kf.new_input(0, u);

    // Take some measurement as an observation:
    double_t z = 2.0;
    // Pass observation into the filter.
    // The index specifies which observer the observation is for.
    kf.new_observation(0, z);

    // Run the filter predict/update iteration.
    kf.iterate();

    // You may grab the current state and covariance estimates from the filter at any time:
    const Eigen::VectorXd& estimated_state = kf.state();
    const Eigen::MatrixXd& estimated_covariance = kf.covariance();
}
```

### 2.2: Unscented Kalman Filter (UKF)

The Unscented Kalman Filter (UKF) can be used for state estimation of nonlinear systems with additive noise.

The UKF library requires the user to extend a base `ukf_t` class to provide state transition and observation functions. The user's `state_transition(xp,x)` and `observation(x,z)` may pull additional information from the extended class's data members during calculation, for example control inputs or a dt. **NOTE** It is critical that these functions must not modify any external data.

The following code snippet demonstrates a very minimal example of how to use the UKF library. More UKF-specific examples can be found under [kalman_filter_examples](https://github.com/pcdangio/ros-kalman_filter_examples/tree/main/src/ukf).

```cpp
#include <kalman_filter/ukf.hpp>

// Create extension of ukf_t to incorporate model dynamics.
class model_t
    : public kalman_filter::ukf_t
{
public:
    // Set up with 2 variables and 1 observer.
    // You may choose however many variables and observers you like.
    model_t()
        : ukf_t(2,1)
    {}

    // OPTIONAL: Stores the current control input.
    double_t u;

private:
    // Implement/override the UKF's state transition model.
    void state_transition(const Eigen::VectorXd& xp, Eigen::VectorXd& x) const override
    {
        // Write your state transition model here.

        // For example:
        x(0) = std::cos(xp(1));
        x(1) = u;
    }
    // Implement/override the UKFs observation model.
    void observation(const Eigen::VectorXd& x, Eigen::VectorXd& z) const override
    {
        // Write your observation model here.

        // For example:
        z(0) = x(1);
    }
};


int32_t main(int32_t argc, char** argv)
{
    model_t ukf;

    // Set up process and measurement covariance matrices.
    ukf.Q(0,0) = 0.1;
    ukf.Q(1,1) = 0.1;
    ukf.R(0,0) = 1;

    // OPTIONAL: Set the initial state and covariance of the model.
    Eigen::VectorXd x_o(2);
    x_o(0) = 0;
    x_o(1) = 0;
    Eigen::MatrixXd P_o(2,2);
    P_o(0,0) = 0.5;
    P_o(1,1) = 0.5;
    ukf.initialize_state(x_o, P_o);

    // The following can be run in a loop:

    // Calculate some new control input and store within the model:
    ukf.u = 5.0;

    // Take some measurement as an observation:
    double_t z = 2.0;
    // Pass observation into the filter.
    // The index specifies which observer the observation is for.
    ukf.new_observation(0, z);

    // Run the filter predict/update iteration.
    ukf.iterate();

    // You may grab the current state and covariance estimates from the filter at any time:
    const Eigen::VectorXd& estimated_state = ukf.state();
    const Eigen::MatrixXd& estimated_covariance = ukf.covariance();
}
```

### 2.3: Unscented Kalman Filter - Augmented (UKFA)

The Unscented Kalman Filter - Augmented (UKFA) can be used for state estimation of nonlinear systems with any type of noise (additive, multiplicative, etc.). The UKFA differs from the UKF in that the process (q) and observation (r) noise parameters are given to the user in the `state_transition` and `observation` functions so that the user may specify their influence on the model. **NOTE:** While the UKFA can handle any type of noise, it is more computationally complex than a standard UKF and takes longer to run.

The UKFA library requires the user to extend a base `ukfa_t` class to provide state transition and observation functions. The user's `state_transition(xp,q,x)` and `observation(x,r,z)` may pull additional information from the extended class's data members during calculation, for example control inputs or a dt. **NOTE:** It is critical that these functions must not modify any external data.

The following code snippet demonstrates a very minimal example of how to use the UKFA library. More UKFA-specific examples can be found under [kalman_filter_examples](https://github.com/pcdangio/ros-kalman_filter_examples/tree/main/src/ukfa).

```cpp
#include <kalman_filter/ukfa.hpp>

// Create extension of ukfa_t to incorporate model dynamics.
class model_t
    : public kalman_filter::ukfa_t
{
public:
    // Set up with 2 variables and 1 observer.
    // You may choose however many variables and observers you like.
    model_t()
        : ukfa_t(2,1)
    {}

    // OPTIONAL: Stores the current control input.
    double_t u;

private:
    // Implement/override the UKF's state transition model.
    void state_transition(const Eigen::VectorXd& xp, const Eigen::VectorXd& q, Eigen::VectorXd& x) const override
    {
        // Write your state transition model here.

        // For example:
        x(0) = std::cos(xp(1)) * q(0);
        x(1) = u + q(1);
    }
    // Implement/override the UKFs observation model.
    void observation(const Eigen::VectorXd& x, const Eigen::VectorXd& r, Eigen::VectorXd& z) const override
    {
        // Write your observation model here.

        // For example:
        z(0) = x(1) + r(0);
    }
};


int32_t main(int32_t argc, char** argv)
{
    model_t ukfa;

    // Set up process and measurement covariance matrices.
    ukfa.Q(0,0) = 0.1;
    ukfa.Q(1,1) = 0.1;
    ukfa.R(0,0) = 1;

    // OPTIONAL: Set the initial state and covariance of the model.
    Eigen::VectorXd x_o(2);
    x_o(0) = 0;
    x_o(1) = 0;
    Eigen::MatrixXd P_o(2,2);
    P_o(0,0) = 0.5;
    P_o(1,1) = 0.5;
    ukfa.initialize_state(x_o, P_o);

    // The following can be run in a loop:

    // Calculate some new control input and store within the model:
    ukfa.u = 5.0;

    // Take some measurement as an observation:
    double_t z = 2.0;
    // Pass observation into the filter.
    // The index specifies which observer the observation is for.
    ukfa.new_observation(0, z);

    // Run the filter predict/update iteration.
    ukfa.iterate();

    // You may grab the current state and covariance estimates from the filter at any time:
    const Eigen::VectorXd& estimated_state = ukfa.state();
    const Eigen::MatrixXd& estimated_covariance = ukfa.covariance();
}
```