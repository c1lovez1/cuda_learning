cmake_minimum_required(VERSION 3.10)
PROJECT(zhifangtu)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

include_directories(${PROJECT_SOURCE_DIR}/include)

cuda_add_executable(1_histogram_cpu src/1_histogram_cpu.cu src/cuda_error_check.cu)
cuda_add_executable(1_histogram_cpu_cpp src/1_histogram_cpu_cpp.cu src/cuda_error_check.cu)
cuda_add_executable(1_histogram_cpu_c src/1_histogram_cpu_c.cu src/cuda_error_check.cu)


cuda_add_executable(2_histogram_globalgpu src/2_histogram_globalgpu.cu src/cuda_error_check.cu)
cuda_add_executable(2_histogram_sharedgpu src/2_histogram_sharedgpu.cu src/cuda_error_check.cu)
cuda_add_executable(2_histogram_gpuThrust src/2_histogram_gpuThrust.cu src/cuda_error_check.cu)
cuda_add_executable(2_histogram_gpuParallelism src/2_histogram_gpuParallelism.cu src/cuda_error_check.cu)
