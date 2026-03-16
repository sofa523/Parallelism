cmake_minimum_required(VERSION 3.10)
project(OpenMP_Integration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")

# Создаём папку для бинарников
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${OUTPUT_DIR})

# Включаем поддержку OpenMP
find_package(OpenMP REQUIRED)

# Ищем библиотеку математических функций
find_package(Threads REQUIRED)

# Добавляем исполняемый файл для задания 2
add_executable(integrate integrate.cpp)

# Устанавливаем выходную директорию
set_target_properties(integrate PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)

# Подключаем OpenMP и математическую библиотеку
target_link_libraries(integrate PRIVATE 
    OpenMP::OpenMP_CXX 
    Threads::Threads
    m
)

# Включаем поддержку OpenMP при компиляции
target_compile_options(integrate PRIVATE ${OpenMP_CXX_FLAGS})

# Сообщение об успешной настройке
message(STATUS "Project configured successfully")
message(STATUS "Output directory: ${OUTPUT_DIR}")
message(STATUS "OpenMP version: ${OpenMP_CXX_VERSION}")