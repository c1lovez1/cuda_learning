#include "cuda_error_check.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
// #include <algorithm>
// #include <bits/stdc++.h>
// 0.37

typedef int INT;
typedef float FLOAT;

template <int blockSize>
__global__ void reduce_v0(float *d_in, float *d_out)
{
    __shared__ float smem[blockSize];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockSize + threadIdx.x;
    smem[tid] = d_in[gtid];
    __syncthreads();

    for (int index = blockDim.x/2; index > 0; index >>= 1)
    {
        if (tid < index)
        {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_out[blockIdx.x] = smem[0];
    }
}

bool CheckResult(float *out, float groudtruth, int n)
{
    float res = 0;
    for (int i = 0; i < n; i++)
    {
        res += out[i];
    }
    return fabs(res - groudtruth) < 1e-5;  // 使用一定的容忍度
    // if (res != groudtruth)
    // {
    //     return false;
    // }
    // return true;
}

int main()
{
    FLOAT milliseconds = 0;
    int N = 25600000;
    int nbyte = N * sizeof(float);

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int blockSize = 256;
    const int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);

    float *a = (float *)malloc(nbyte);
    float *d_a;
    cudaMalloc((void **)&d_a, nbyte);

    float *out = (float *)malloc(GridSize * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, GridSize * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        a[i] = 1.0f;
    }

    cudaMemcpy(d_a, a, nbyte, cudaMemcpyHostToDevice);

    float groudtruth = N * 1.0f;

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_v0<blockSize><<<Grid, Block>>>(d_a, d_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);

    bool is_right = CheckResult(out, groudtruth, GridSize);

    if (is_right)
    {
        printf("the ans is right\n");
    }
    else
    {
        printf("the ans is wrong\n");
        // for(int i = 0; i < GridSize;i++){
        // printf("res per block : %lf ",out[i]);
        //}
        // printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_v0 latency = %f ms\n", milliseconds);

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_a);
    cudaFree(d_out);

    free(a);
    free(out);

    return 0;
}