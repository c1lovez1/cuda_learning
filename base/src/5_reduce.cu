#include "cuda_error_check.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

typedef float FLOAT;

__global__ void reduce_baseline(const int *input, int *output, size_t N)
{
    int sum = 0;
    // size_t idx = blockIdx.x*blockDim.x+threadIdx.x;
    for (size_t i = 0; i < N; i++)
    {
        sum += input[i];
    }
    *output = sum;
}

bool CheckResult(int *out, int groudtruth, int N)
{
    if (*out != groudtruth)
    {
        return false;
    }
    return true;
}

int main()
{
    FLOAT milliseconds = 0;
    const size_t N = 25600000;

    const int blockSize = 1;
    const int GridSize = 1;


    // // 使用统一内存分配
    // int *a;
    // cudaMallocManaged(&a, N * sizeof(int));
    // int *out;
    // cudaMallocManaged(&out, GridSize * sizeof(int));

    int *a = (int *)malloc(N * sizeof(int));
    int *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(int));
    int *out = (int *)malloc(GridSize * sizeof(int));
    int *d_out;
    cudaMalloc((void **)&d_out, GridSize * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        a[i] = 1;
    }

    FLOAT groudtruth = N * 1;

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_baseline<<<Grid, Block>>>(d_a, d_out, N);
    // reduce_baseline<<<Grid, Block>>>(a, out, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // 同步设备到主机内存（必要时）
    cudaDeviceSynchronize();

    bool is_right = CheckResult(out, groudtruth, GridSize);
    if (is_right)
    {
        printf("the ans is right\n");
    }
    else
    {
        printf("the ans is wrong\n");
        for (int i = 0; i < GridSize; i++)
        {
            printf("res per block : %d ", out[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_baseline latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // cudaFree(a);
    // cudaFree(out);

    return 0;
}
