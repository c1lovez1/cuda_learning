#include "cuda_error_check.h"
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>

#define NUM_BINS 256
#define DATA_SIZE 2560000
#define BLOCK_SIZE 256

__global__ void histParallelism(const unsigned char *d_buffer, const int N, unsigned int *d_histo) {
    int tid = threadIdx.x;
    int n = tid + blockDim.x * blockIdx.x;
    __shared__ unsigned int s_histo[NUM_BINS];

    // 初始化共享内存直方图
    if (tid < NUM_BINS) {
        s_histo[tid] = 0;
    }
    __syncthreads();

    // 计算直方图
    const int offset = gridDim.x * blockDim.x;
    while (n < N) {
        atomicAdd(&s_histo[d_buffer[n]], 1);
        n += offset;
    }
    __syncthreads();

    // 将共享内存直方图累加到全局直方图
    if (tid < NUM_BINS) {
        atomicAdd(&d_histo[tid], s_histo[tid]);
    }
}

bool CheckResult(unsigned int *out, int* groundtruth, int N){
    for (int i = 0; i < N; i++){
        if (out[i] != groundtruth[i]) {
            printf("in checkres, out[i]=%d, gt[i]=%d\n", out[i], groundtruth[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    unsigned char h_data[DATA_SIZE];
    for (int i = 0; i < DATA_SIZE; i++){
        h_data[i] = i % NUM_BINS;
    }

    unsigned int h_hist[NUM_BINS] = {0};

    unsigned char *d_data;
    unsigned int  *d_hist;

    cudaMalloc((void**)&d_data, DATA_SIZE * sizeof(unsigned char));
    cudaMalloc((void**)&d_hist, NUM_BINS * sizeof(unsigned int));

    cudaMemcpy(d_data, h_data, DATA_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist, h_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);

    const int BlockperGrid = (DATA_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *groundtruth = (int *)malloc(NUM_BINS * sizeof(int));
    for (int j = 0; j < NUM_BINS; j++){
        groundtruth[j] = DATA_SIZE / NUM_BINS;
    }

    float milliseconds = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    histParallelism<<<BlockperGrid, BLOCK_SIZE>>>(d_data, DATA_SIZE, d_hist);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    bool is_right = CheckResult(h_hist, groundtruth, NUM_BINS);
    if (is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for (int i = 0; i < NUM_BINS; i++){
            printf("%d ", h_hist[i]);
        }
        printf("\n");
    }
    printf("histogram + shared_mem + multi_value latency = %f ms\n", milliseconds);

    // 清理CUDA事件对象
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_data);
    cudaFree(d_hist);
    free(groundtruth);

    return 0;
}
