// #include <iostream>
// #include <cuda.h>
// #include <cuda_runtime.h>

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_error_check.h"  // 包含错误检查头文件

__global__ void hello_cuda() {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("block id = [ %d ], thread id = [ %d ] hello cuda\n", blockIdx.x, idx);
}

int main() {
    // 使用错误检查宏来启动CUDA内核
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    hello_cuda<<<1, 1>>>();
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // std::cout<<"test"<<std::endl;
    return 0;
}
