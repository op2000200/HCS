cmake_minimum_required(VERSION 3.24...3.30)

project(demo VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -O3")

add_executable(demo_exec
    ${CMAKE_CURRENT_SOURCE_DIR}/demo.cu
)

target_include_directories(demo_exec PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)