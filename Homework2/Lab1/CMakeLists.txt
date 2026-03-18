cmake_minimum_required(VERSION 3.10)
project(MatrixVectorProduct LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")

add_executable(matrix_vector matrix_vector.cpp)