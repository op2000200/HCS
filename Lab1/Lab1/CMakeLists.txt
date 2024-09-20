cmake_minimum_required(VERSION 3.24...3.30)

project(cuda-sample1 VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -O3")

include(GoogleTest)
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)

FetchContent_MakeAvailable(googletest)

add_subdirectory(inc)

add_executable(cuda_executable
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu
)

add_executable(test_exec
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cu
)

target_include_directories(test_exec PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/include
)

target_link_libraries(test_exec PUBLIC
    GTest::gtest_main
)

enable_testing()
gtest_discover_tests(test_exec)