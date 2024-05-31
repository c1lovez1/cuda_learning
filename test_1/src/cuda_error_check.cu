#include "cuda_error_check.h"

void CheckCudaError (cudaError_t result,const char* const func,const char* const file,int const line){
    if (result != cudaSuccess){
        fprintf(stderr,"Cuda error at %s:%d code = %d(%s) \"%s\" \n",
            file,line,static_cast<unsigned int>(result),cudaGetErrorName(result),func);
        cudaDeviceReset;
        exit(EXIT_FAILURE);
    }
}

void CheckCuBilasStatus(cudaError_t result,const char* const func,const char* const file,int const line){
    if (result != cudaSuccess){
        fprintf(stderr,"Cuda error at %s:%d code = %d(%s) \"%s\"",file,line,static_cast<unsigned int>(result),cudaGetErrorName(result),func);
    }
    cudaDeviceReset;
    exit(EXIT_FAILURE);
}
