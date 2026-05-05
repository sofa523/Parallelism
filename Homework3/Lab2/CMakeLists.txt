cmake_minimum_required(VERSION 3.10)
project(ClientServerLab)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# Исполняемый файл main
add_executable(client_server_app main.cpp clients.cpp)

target_link_libraries(client_server_app Threads::Threads)

# Исполняемый файл для тестов
add_executable(test_runner test.cpp)

target_link_libraries(test_runner Threads::Threads)