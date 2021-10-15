/// \file kalman_filter/base.hpp
/// \brief Defines the kalman_filter::base_t class.
#ifndef KALMAN_FILTER___BASE_H
#define KALMAN_FILTER___BASE_H

#include <eigen3/Eigen/Dense>

#include <map>
#include <fstream>

/// \brief Contains objects for Kalman Filtering.
namespace kalman_filter {

/// \brief Provides base functionality for all Kalman Filter object types.
class base_t
{
public:
    // CONSTRUCTORS
    /// \brief Instantiates a new base_t object.
    /// \param n_variables The number of variables in the state vector.
    /// \param n_observers The number of state observers.
    base_t(uint32_t n_variables, uint32_t n_observers);
    ~base_t();

    // FILTER METHODS
    /// \brief Predicts a new state and performs update corrections with available observations.
    /// \note The iteration rate should be at least as fast as the fastest observer rate.
    virtual void iterate() = 0;
    /// \brief Adds a new observation to the filter.
    /// \param observer_index The index of the observer that made the observation.
    /// \param observation The value of the observation.
    void new_observation(uint32_t observer_index, double_t observation);
    /// \brief Indicates if a new observation is available.
    /// \param observer_index The index of the observer to check for a new observation.
    /// \returns TRUE if a new observation is available, otherwise FALSE.
    bool has_observation(uint32_t observer_index) const;

    // ACCESS
    /// \brief Gets the number of variables in the state vector.
    /// \returns The number of variables.
    uint32_t n_variables() const;
    /// \brief Gets the number of observers.
    /// \returns The number of observers.
    uint32_t n_observers() const;
    /// \brief Gets the current estimated value of a state variable.
    /// \param index The index of the variable to get.
    /// \returns The current estimated value of the state variable.
    double_t state(uint32_t index) const;
    /// \brief Sets the value of an estimated state variable.
    /// \param index The index of the variable to set.
    /// \param value The value to assign to the variable.
    void set_state(uint32_t index, double_t value);
    /// \brief Gets the current covariance between two estimated state variables.
    /// \param index_a The index of the first estimated state.
    /// \param index_b The index of the second estimated state.
    /// \returns The covariance between the two estimated states.
    double_t covariance(uint32_t index_a, uint32_t index_b) const;
    /// \brief Sets the covariance between two estimated state variables.
    /// \param index_a The index of the first estimated state.
    /// \param index_b The index of the second estimated state.
    /// \param value The value to assign to the covariance.
    void set_covariance(uint32_t index_a, uint32_t index_b, double_t value);

    // COVARIANCES
    /// \brief The process noise covariance matrix.
    Eigen::MatrixXd Q;
    /// \brief The observation noise covariance matrix.
    Eigen::MatrixXd R;

    // LOGGING
    /// \brief Opens up a log file and begins logging data.
    /// \param log_file The file to log to.
    /// \param precision The precision to write numbers with. DEFAULT = 6
    /// \returns TRUE if the logging successfully started, otherwise FALSE.
    bool start_log(const std::string& log_file, uint8_t precision = 6);
    /// \brief Stops logging.
    void stop_log();

protected:
    
    // DIMENSIONS
    /// \brief The number of variables being estimated by the system.
    uint32_t n_x;
    /// \brief The number of observers.
    uint32_t n_z;

    // STORAGE: PREDICTION
    /// \brief The variable vector.
    Eigen::VectorXd x;
    /// \brief The variable covariance matrix.
    Eigen::MatrixXd P;

    // STORAGE: UPDATE
    /// \brief The predicted observation vector.
    Eigen::VectorXd z;
    /// \brief The predicted observation covariance.
    Eigen::MatrixXd S;
    /// \brief The innovation cross covariance.
    Eigen::MatrixXd C;

    // STORAGE: TEMPORARIES
    /// \brief A temporary of size n_x,n_x.
    Eigen::MatrixXd t_xx;

    // METHODS
    /// \brief Indicates if any observations have been made since the last iteration.
    /// \returns TRUE if new observations exist, otherwise FALSE.
    bool has_observations() const;
    /// \brief Performs a Kalman update masked by available observations.
    /// \details S and C must be calculated first.
    void masked_kalman_update();
    /// \brief Writes the predicted state to the log file.
    void log_predicted_state();
    /// \brief Writes observations to the log file.
    /// \param empty Indicates if there are no observations available.
    void log_observations(bool empty = false);
    /// \brief Writes the estimated state to the log file.
    void log_estimated_state();

private:
    // VARIABLES
    /// \brief Stores the actual observations made between iterations.
    std::map<uint32_t, double_t> m_observations;

    // LOGGING
    /// \brief The log file instance.
    std::ofstream m_log_file;
};

}

#endif