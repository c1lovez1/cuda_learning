// cuda_error_check.h

#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H


#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

// 定义一个宏来检查CUDA API调用是否有错误
#define CHECK_CUDA_ERROR(call) checkCudaError( (call), #call, __FILE__, __LINE__ )
#define CHECK_CUBLAS_STATUS(call) checkCublasStatus( (call), #call, __FILE__, __LINE__ )

// 错误检查函数的声明
void checkCudaError(cudaError_t result, char const *const func, const char *const file, int const line);
void checkCublasStatus(cublasStatus_t result, char const *const func, const char *const file, int const line);

#endif // CUDA_ERROR_CHECK_H
