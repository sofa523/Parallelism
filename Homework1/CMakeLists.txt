cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(test VERSION 0.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Переменная для выбора типа (по умолчанию double)
option(use_float "Использовать тип float вместо double" OFF)

if(use_float)
    add_compile_definitions(ARRAY_TYPE=1)
else()
    add_compile_definitions(ARRAY_TYPE=0)
endif()

add_compile_definitions(array_type=${array_type})
add_executable(test Homework.cpp)