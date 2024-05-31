#include "cuda_error_check.h"
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>

#define NUM_BINS 256
#define DATA_SIZE 2560000

template <int ThreadperBlock>
__global__ void histshared(const unsigned char *d_data,unsigned int *d_hist,int N){
    __shared__ unsigned int s_hist[NUM_BINS];
    int tid = threadIdx.x;
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < NUM_BINS){
        s_hist[tid] = 0;
    }
    __syncthreads();

    if (gtid < N){
        atomicAdd(&s_hist[d_data[gtid]],1);
    }
    __syncthreads();

    atomicAdd(&d_hist[tid], s_hist[tid]);
}

bool CheckResult(unsigned int *out, int* groudtruth, int N){
    for (int i = 0; i < N; i++){
        if (out[i] != groudtruth[i]) {
            printf("in checkres, out[i]=%d, gt[i]=%d\n", out[i], groudtruth[i]);
            return false;
        }
    }
    return true;
}


int main()
{
    unsigned char h_data[DATA_SIZE];
    for (int i =0;i<DATA_SIZE;i++){
        h_data[i] = i%NUM_BINS;
    }

    unsigned int h_hist[NUM_BINS] = {0};

    unsigned char *d_data;
    unsigned int  *d_hist;

    cudaMalloc((void**)&d_data,DATA_SIZE*sizeof(unsigned char));
    cudaMalloc((void**)&d_hist,NUM_BINS*sizeof(unsigned int));

    cudaMemcpy(d_data,h_data,DATA_SIZE*sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist,h_hist,NUM_BINS*sizeof(unsigned int),cudaMemcpyHostToDevice);

    const int ThreadperBlock = 256;
    const int BlockperGrid   = (DATA_SIZE + ThreadperBlock -1)/ThreadperBlock;

    int *groudtruth = (int *)malloc(256 * sizeof(int));;
    for(int j = 0; j < 256; j++){
        groudtruth[j] = 10000;
    }


    float milliseconds = 0.0f;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    histshared<ThreadperBlock><<<BlockperGrid,ThreadperBlock>>>(d_data,d_hist,DATA_SIZE);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    
     cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);  // 从设备端拷贝回主机端

    bool is_right = CheckResult(h_hist, groudtruth, 256);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < 256; i++){
            printf("%d ", h_hist[i]);
        }
        printf("\n");
    }
    printf("histogram + shared_mem + multi_value latency = %f ms\n", milliseconds);    


    // 清理CUDA事件对象
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    return 0;
}