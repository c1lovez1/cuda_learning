cmake_minimum_required(VERSION 3.10)
project(cuda_base)

# 设置 CUDA 编译选项
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 添加可执行文件
cuda_add_executable(1_hello_cuda src/1_helloworld.cu src/cuda_error_check.cu)

# 添加可执行文件
cuda_add_executable(2_vectorAdd src/2_vectorAdd.cu src/cuda_error_check.cu)

# 添加可执行文件
cuda_add_executable(2.1_vectorAdd src/2.1_vectorAdd.cu src/cuda_error_check.cu)


# 添加可执行文件
cuda_add_executable(5_reduce src/5_reduce.cu src/cuda_error_check.cu)

# 添加可执行文件
cuda_add_executable(5_reduce_v0 src/5_reduce_v0.cu src/cuda_error_check.cu)

# 添加可执行文件
cuda_add_executable(5_reduce_v1 src/5_reduce_v1.cu src/cuda_error_check.cu)

# 添加可执行文件
cuda_add_executable(5_reduce_v2 src/5_reduce_v2.cu src/cuda_error_check.cu)

# 添加可执行文件
cuda_add_executable(5_reduce_v3 src/5_reduce_v3.cu src/cuda_error_check.cu)

# 添加可执行文件
cuda_add_executable(5_reduce_v4 src/5_reduce_v4.cu src/cuda_error_check.cu)

# 添加可执行文件
cuda_add_executable(5_reduce_v5 src/5_reduce_v5.cu src/cuda_error_check.cu)

# 添加可执行文件
cuda_add_executable(5_reduce_v6 src/5_reduce_v6.cu src/cuda_error_check.cu)

# 添加可执行文件
cuda_add_executable(6_warp_level_reduce src/6_warp_level_reduce.cu src/cuda_error_check.cu)


# # 添加所有的源文件
# file(GLOB SOURCES "src/*.cu")

# # 添加可执行文件
# cuda_add_executable(hello_cuda ${SOURCES})
