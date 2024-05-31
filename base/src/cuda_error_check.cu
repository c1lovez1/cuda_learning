// cuda_error_check.cu

#include "cuda_error_check.h"

void checkCudaError(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        // 在发生错误时终止程序
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

void checkCublasStatus(cublasStatus_t result, char const *const func, const char *const file, int const line) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d code=%d \"%s\" \n",
                file, line, static_cast<unsigned int>(result), func);
        // 在发生错误时终止程序
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
