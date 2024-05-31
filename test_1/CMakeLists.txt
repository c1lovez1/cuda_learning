cmake_minimum_required(VERSION 3.10)
PROJECT(TEST_1)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}-std=c++11")

include_directories(${PROJECT_SOURCE_DIR}/include)

cuda_add_executable(reduce src/reduce.cu src/cuda_error_check.cu)
cuda_add_executable(histogram src/histogram.cu src/cuda_error_check.cu)