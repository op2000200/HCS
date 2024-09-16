cmake_minimum_required(VERSION 3.24...3.30)

project(cuda-sample1 VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
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

add_subdirectory(cuda-sample1-lib)

add_executable(cuda-sample1-test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/tests.cu
)

target_include_directories(cuda-sample1-test PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda-sample1-lib/include
)

target_link_libraries(cuda-sample1-test PUBLIC
    cuda-sample1-lib
    GTest::gtest_main
)

enable_testing()
gtest_discover_tests(cuda-sample1-test)
