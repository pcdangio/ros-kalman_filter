cmake_minimum_required(VERSION 3.0.2)
project(kalman_filter)

# Find catkin and catkin package dependencies.
find_package(catkin REQUIRED)

# Find Eigen system dependency.
find_package(Eigen3 REQUIRED)

# Set up catkin package.
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}_kf ${PROJECT_NAME}_ukf ${PROJECT_NAME}_ukfa
  DEPENDS EIGEN3)

# Set up include directories.
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS})

# Build KF library.
add_library(${PROJECT_NAME}_kf
  src/kalman_filter/base.cpp
  src/kalman_filter/kf.cpp)

# Build UKF library.
add_library(${PROJECT_NAME}_ukf
  src/kalman_filter/base.cpp
  src/kalman_filter/ukf.cpp)

# Build UKFA library.
add_library(${PROJECT_NAME}_ukfa
  src/kalman_filter/base.cpp
  src/kalman_filter/ukfa.cpp)

# Install libraries.
install(TARGETS ${PROJECT_NAME}_kf ${PROJECT_NAME}_ukf ${PROJECT_NAME}_ukfa
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp")