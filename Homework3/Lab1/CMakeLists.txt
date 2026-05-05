cmake_minimum_required(VERSION 3.10)
project(MatrixVectorProduct LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matrix_vector matrix_vector.cpp)