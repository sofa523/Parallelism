cmake_minimum_required(VERSION 3.10)
project(MatrixVectorProduct LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_executable(matrix_vector matrix_vector.cpp)
target_link_libraries(matrix_vector PRIVATE OpenMP::OpenMP_CXX Threads::Threads)