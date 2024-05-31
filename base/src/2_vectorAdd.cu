#include "cuda_error_check.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

typedef float FLOAT;

__global__ void vec_add(FLOAT *X,FLOAT *Y,FLOAT *Z,int N){
    int idx = (blockDim.x*(blockIdx.y * gridDim.x+blockIdx.x)+threadIdx.x);
    if (idx<N){
        Z[idx] = X[idx]+Y[idx]; 
    }
}

void vec_add_cpu(FLOAT *X,FLOAT *Y,FLOAT *Z,int N){
    for (int i =0;i<N;i++){
        Z[i] = X[i] + Y[i];
    }
}


int main()
{
    int N = 1000000;
    int nbytes = N * sizeof(float);

    int bs = 256;

    // 手动向上取整
    int s = ceil(sqrt(N + bs - 1. / bs));
    dim3 grid(s, s);

    FLOAT *hx, *dx;
    FLOAT *hy, *dy;
    FLOAT *hz, *dz;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&dx, nbytes));
    cudaMalloc((void **)&dy, nbytes);
    cudaMalloc((void **)&dz, nbytes);

    FLOAT millisecond = 0;

    hx = (FLOAT *)malloc(nbytes);
    hy = (FLOAT *)malloc(nbytes);
    hz = (FLOAT *)malloc(nbytes);

    for (int i = 0; i < N; i++)
    {
        hx[i] = 1;
        hy[i] = 1;
    }

    cudaMemcpy(dx,hx,nbytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dy,hy,nbytes,cudaMemcpyHostToDevice);

    cudaEvent_t start,stop;

    cudaEventCreate(&start);// 创建事件
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vec_add<<<grid,bs>>>(dx,dy,dz,N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecond,start,stop);
    
    cudaMemcpy(hz,dz,nbytes,cudaMemcpyDeviceToHost);

    FLOAT *hz_cpu = (FLOAT*) malloc(nbytes);
    vec_add_cpu(hx,hy,hz_cpu,N);

    for (int i =0;i<N;i++){
        if ((fabs(hz_cpu[i]-hz[i]))>1e-6){
            printf("Result verification failed at element index %d!\n", i);
        }
    }

    printf("Result right\n");
    std::cout<<"消耗时间："<<millisecond<<"ms"<<std::endl;

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);
    free(hz_cpu);

    return 0;
}