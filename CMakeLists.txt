cmake_minimum_required(VERSION 3.10)

project(SLAM_For_Robotics)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(matplotplusplus)
add_subdirectory(eigen-3.4.0)

add_executable(homogeneous_transform src/homogeneous_transform.cpp)
target_link_libraries(homogeneous_transform PUBLIC matplot)
target_link_libraries(homogeneous_transform PUBLIC eigen)