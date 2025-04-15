cmake_minimum_required(VERSION 3.10)
set(CMAKE_GENERATOR_PLATFORM x64)

project(SLAM_For_Robotics)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/matplotplusplus)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/eigen-3.4.0)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11)

add_executable(homogeneous_transform src/homogeneous_transform/homogeneous_transform.cpp)
target_link_libraries(homogeneous_transform PUBLIC matplot)
target_link_libraries(homogeneous_transform PUBLIC eigen)

pybind11_add_module (
    velocity_motion_model 
    src/Velocity_Motion_Model/velocity_motion_model.cpp
)
target_link_libraries(velocity_motion_model PUBLIC eigen)

pybind11_add_module (
    odometry_motion_model 
    src/Odometry_Motion_Model/odometry_motion_model.cpp
)
target_link_libraries(odometry_motion_model PUBLIC eigen)