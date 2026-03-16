cmake_minimum_required(VERSION 3.10)
project(OpenMP_SimpleIteration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")

# Создаём папку для бинарников
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${OUTPUT_DIR})

# Включаем поддержку OpenMP
find_package(OpenMP REQUIRED)

# Ищем библиотеку потоков и математическую библиотеку
find_package(Threads REQUIRED)

# Добавляем исполняемый файл для задания 3
add_executable(simple_iteration simple_iteration.cpp)

# Устанавливаем выходную директорию
set_target_properties(simple_iteration PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)

# Подключаем OpenMP и математическую библиотеку
target_link_libraries(simple_iteration PRIVATE 
    OpenMP::OpenMP_CXX 
    Threads::Threads
    m
)

# Включаем поддержку OpenMP при компиляции
target_compile_options(simple_iteration PRIVATE ${OpenMP_CXX_FLAGS})

# Сообщение об успешной настройке
message(STATUS "Project configured successfully")
message(STATUS "Output directory: ${OUTPUT_DIR}")
message(STATUS "OpenMP version: ${OpenMP_CXX_VERSION}")