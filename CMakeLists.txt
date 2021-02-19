cmake_minimum_required(VERSION 3.0.2)
project(kalman_filter)

# Compile as C++11, supported in ROS Kinetic and newer.
add_compile_options(-std=c++11)

# Find catkin and catkin package dependencies.
find_package(catkin REQUIRED COMPONENTS
  roscpp)

# Find Eigen system dependency.
find_package(Eigen3 REQUIRED)

# Set up catkin package.
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}_ukf
  CATKIN_DEPENDS roscpp
  DEPENDS EIGEN3
)

# Set up include directories.
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS})

# Build UKF library.
add_library(${PROJECT_NAME}_ukf
  src/kalman_filter/base.cpp
  src/kalman_filter/ukf.cpp)

# Install UKF library.
install(TARGETS ${PROJECT_NAME}_ukf
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h")