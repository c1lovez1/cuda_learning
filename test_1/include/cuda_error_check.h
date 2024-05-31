#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H

#include <cuda_runtime.h>
#include <cublas.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(call) CheckCudaError((call),#call,__FILE__,__LINE__);
#define CHECK_CUBILAS__STATUS(call) CheckCuBilasStatus((call),#call,__FILE__,__LINE__);

void CheckCudaError(cudaError_t result,const char*const func,const char*const file,int const line);
void CheckCuBilasStatus(cudaError_t result,const char*const func,const char*const file,int const line);

#endif // CUDA_ERROR_CHECK_H