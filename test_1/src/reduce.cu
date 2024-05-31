#include "cuda_error_check.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdlib>
#include <cmath>

#define ThreadperBlock  256

typedef float Datatype;

void reduce_cpu(Datatype* h_data, Datatype* sum_cpu, int N) {
    for (int i = 0; i < N; i++) {
        *sum_cpu += h_data[i];
    }
}

// 1.8ms
__global__ void reduce_baseline(const Datatype* d_data, Datatype* d_sum, unsigned int N) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        atomicAdd(d_sum, d_data[idx]);
    }
}


// 0.47ms
__global__ void reduce_shared(const Datatype* d_data, Datatype* d_sum, unsigned int N) { 
    unsigned int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid  = threadIdx.x;
    __shared__ Datatype smem[ThreadperBlock];

    smem[tid] = gtid<N ? d_data[gtid] : 0.0;
    __syncthreads();

    for (int index = ThreadperBlock>>1; index>0; index >>= 1){
        if (tid < index){
            smem[tid] += smem[tid+index];  
        }
        __syncthreads();
    }

    if (tid == 0){
        atomicAdd(d_sum,smem[0]);
    }
}




__global__ void reduce_Parallelism(const Datatype* d_data, Datatype* d_sum, unsigned int N) { 
    unsigned int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid  = threadIdx.x;
    __shared__ Datatype smem[ThreadperBlock];

    Datatype sum = 0.0f;
    for (; gtid < N ;gtid += blockDim.x * gridDim.x){
        sum += d_data[gtid];
    }
    smem[tid] = sum;
    __syncthreads();

    for (int index = ThreadperBlock>>1; index>0; index >>= 1){
        if (tid < index){
            smem[tid] += smem[tid+index];  
        }
        __syncthreads();
    }

    if (tid == 0){
        atomicAdd(d_sum,smem[0]);
    }
}


__global__ void reduce_Parallelism_unroll(const Datatype* d_data, Datatype* d_sum, unsigned int N) { 
    unsigned int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid  = threadIdx.x;
    __shared__ Datatype smem[ThreadperBlock];

    Datatype sum = 0.0f;
    for (; gtid < N ;gtid += blockDim.x * gridDim.x){
        sum += d_data[gtid];
    }
    smem[tid] = sum;
    __syncthreads();

    # pragma unroll
    for (int index = ThreadperBlock>>1; index>0; index >>= 1){
        if (tid < index){
            smem[tid] += smem[tid+index];  
        }
        __syncthreads();
    }

    if (tid == 0){
        atomicAdd(d_sum,smem[0]);
    }
}


__global__ void reduce_Warp_unroll (const Datatype* d_data, Datatype* d_sum, unsigned int N) { 
    unsigned int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid  = threadIdx.x;
    __shared__ Datatype smem[ThreadperBlock];

    smem[tid] = gtid<N ? d_data[gtid] : 0.0;
    __syncthreads();

    # pragma unroll
    for (int index = ThreadperBlock>>1; index>=32; index >>= 1){
        if (tid < index){
            smem[tid] += smem[tid+index];  
        }
        __syncthreads();
    }

    # pragma unroll
    for (int index = 16;index>0;index>>=1){
        if (tid < index){
            smem[tid] += smem[tid+index];
        }
        __syncwarp();
    }

    if (tid == 0){
        atomicAdd(d_sum,smem[0]);
    }
}


__global__ void reduce_Warp(const Datatype* d_data, Datatype* d_sum, unsigned int N) { 
    unsigned int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid  = threadIdx.x;
    __shared__ Datatype smem[ThreadperBlock];

    smem[tid] = gtid<N ? d_data[gtid] : 0.0;
    __syncthreads();

    for (int index = ThreadperBlock>>1; index>=32; index >>= 1){
        if (tid < index){
            smem[tid] += smem[tid+index];  
        }
        __syncthreads();
    }

    for (int index = 16;index>0;index>>=1){
        if (tid < index){
            smem[tid] += smem[tid+index];
        }
        __syncwarp();
    }

    if (tid == 0){
        atomicAdd(d_sum,smem[0]);
    }
}


__global__ void reduce_shfl(const Datatype* d_data, Datatype* d_sum, unsigned int N) { 
    unsigned int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid  = threadIdx.x;
    __shared__ Datatype smem[ThreadperBlock];

    smem[tid] = gtid<N ? d_data[gtid] : 0.0;
    __syncthreads();

    for (int index = ThreadperBlock>>1; index>=32; index >>= 1){
        if (tid < index){
            smem[tid] += smem[tid+index];  
        }
        __syncthreads();
    }

    Datatype y = smem[tid];
    for (int index = 16;index>0;index>>=1){
        // smem[tid] += __shfl_down_sync(0xffffffff,smem[tid],index);
        y += __shfl_down_sync(0xffffffff,y,index);
    }

    if (tid == 0){
        // atomicAdd(d_sum,smem[0]);
        atomicAdd(d_sum,y);
    }
}


__global__ void reduce_shfl_unroll(const Datatype* d_data, Datatype* d_sum, unsigned int N) { 
    unsigned int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid  = threadIdx.x;
    __shared__ Datatype smem[ThreadperBlock];

    smem[tid] = gtid<N ? d_data[gtid] : 0.0;
    __syncthreads();

    # pragma unroll
    for (int index = ThreadperBlock>>1; index>=32; index >>= 1){
        if (tid < index){
            smem[tid] += smem[tid+index];  
        }
        __syncthreads();
    }

    Datatype y = smem[tid];
    # pragma unroll
    for (int index = 16;index>0;index>>=1){
        // smem[tid] += __shfl_down_sync(0xffffffff,smem[tid],index);
        y += __shfl_down_sync(0xffffffff,y,index);
    }

    if (tid == 0){
        // atomicAdd(d_sum,smem[0]);
        atomicAdd(d_sum,y);
    }
}







bool CheckResult(Datatype h_sum, Datatype groundtruth) {
    return fabs(h_sum - groundtruth) < 1e-2;
}

int main() {
    float milliseconds = 0.0f;
    int N = 1023 * 1011;
    Datatype h_data[N];
    Datatype h_sum = 0.0f;
    Datatype sum_cpu = 0.0f;

    for (int i = 0; i < N; i++) {
        h_data[i] = 1; // 如果 h_data[i] = 1.0f 则可以返回正确的答案
    }

    reduce_cpu(h_data, &sum_cpu, N);

    Datatype* d_data;
    Datatype* d_sum;
    cudaMalloc((void**)&d_data, N * sizeof(Datatype));
    cudaMalloc((void**)&d_sum, sizeof(Datatype));

    // const int ThreadperBlock = 256;
    const int BlockperGrid = (N + ThreadperBlock - 1) / ThreadperBlock;

    cudaMemcpy(d_data, h_data, N * sizeof(Datatype), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(Datatype));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // reduce_baseline<<<BlockperGrid, ThreadperBlock>>>(d_data, d_sum, N);
    reduce_shared<<<BlockperGrid, ThreadperBlock>>>(d_data, d_sum, N);
    // reduce_Parallelism<<<BlockperGrid, ThreadperBlock>>>(d_data, d_sum, N);
    // reduce_Parallelism_unroll<<<BlockperGrid, ThreadperBlock>>>(d_data, d_sum, N);
    // reduce_Warp<<<BlockperGrid, ThreadperBlock>>>(d_data, d_sum, N);
    // reduce_Warp_unroll<<<BlockperGrid, ThreadperBlock>>>(d_data, d_sum, N);
    // reduce_shfl<<<BlockperGrid, ThreadperBlock>>>(d_data, d_sum, N);
    // reduce_shfl_unroll<<<BlockperGrid, ThreadperBlock>>>(d_data, d_sum, N);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(&h_sum, d_sum, sizeof(Datatype), cudaMemcpyDeviceToHost);

    bool is_right = CheckResult(h_sum, sum_cpu);
    if (is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }

    std::cout << "reduce_baseline latency = " << milliseconds << "ms" << std::endl;
    std::cout << "CPU sum = " << sum_cpu << std::endl;
    std::cout << "GPU sum = " << h_sum << std::endl;
    std::cout << "CPU - GPU = " << sum_cpu-h_sum << std::endl;

    // 释放内存
    cudaFree(d_data);
    cudaFree(d_sum);

    // 释放 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}




// #include <iostream>
// #include <cuda_runtime.h>
// #include <cuda.h>
// #include <cstdlib>
// #include <cmath>

// #define ThreadperBlock  256

// typedef float Datatype;

// void reduce_cpu(Datatype* h_data, Datatype* sum_cpu, int N) {
//     for (int i = 0; i < N; i++) {
//         *sum_cpu += h_data[i];
//     }
// }

// __global__ void reduce_baseline(const Datatype* d_data, Datatype* d_sum, unsigned int N) {
//     unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
//     if (idx < N) {
//         atomicAdd(d_sum, d_data[idx]);
//     }
// }

// __global__ void reduce_shared(const Datatype* d_data, Datatype* d_sum, unsigned int N) { 
//     unsigned int gtid = threadIdx.x + blockDim.x * blockIdx.x;
//     unsigned int tid  = threadIdx.x;
//     __shared__ Datatype smem[ThreadperBlock];

//     smem[tid] = gtid < N ? d_data[gtid] : 0.0;
//     __syncthreads();

//     for (int index = ThreadperBlock >> 1; index > 0; index >>= 1){
//         if (tid < index){
//             smem[tid] += smem[tid + index];  
//         }
//         __syncthreads();
//     }

//     if (tid == 0){
//         atomicAdd(d_sum, smem[0]);
//     }
// }

// __global__ void reduce_Parallelism(const Datatype* d_data, Datatype* d_sum, unsigned int N) { 
//     unsigned int gtid = threadIdx.x + blockDim.x * blockIdx.x;
//     unsigned int tid  = threadIdx.x;
//     __shared__ Datatype smem[ThreadperBlock];

//     Datatype sum = 0.0f;
//     for (; gtid < N; gtid += blockDim.x * gridDim.x){
//         sum += d_data[gtid];
//     }
//     smem[tid] = sum;
//     __syncthreads();

//     for (int index = ThreadperBlock >> 1; index > 0; index >>= 1){
//         if (tid < index){
//             smem[tid] += smem[tid + index];  
//         }
//         __syncthreads();
//     }

//     if (tid == 0){
//         atomicAdd(d_sum, smem[0]);
//     }
// }

// bool CheckResult(Datatype h_sum, Datatype groundtruth) {
//     return fabs(h_sum - groundtruth) < 1e-2;
// }

// int main() {
//     int N = 1023 * 1011;
//     Datatype* h_data = (Datatype*) malloc(N * sizeof(Datatype));
//     Datatype h_sum = 0.0f;
//     Datatype sum_cpu = 0.0f;

//     for (int i = 0; i < N; i++) {
//         h_data[i] = 1; // 如果 h_data[i] = 1.0f 则可以返回正确的答案
//     }

//     reduce_cpu(h_data, &sum_cpu, N);

//     Datatype* d_data;
//     Datatype* d_sum;
//     cudaError_t err;

//     err = cudaMalloc((void**)&d_data, N * sizeof(Datatype));
//     if (err != cudaSuccess) {
//         std::cerr << "CUDA malloc d_data failed: " << cudaGetErrorString(err) << std::endl;
//         return -1;
//     }

//     err = cudaMalloc((void**)&d_sum, sizeof(Datatype));
//     if (err != cudaSuccess) {
//         std::cerr << "CUDA malloc d_sum failed: " << cudaGetErrorString(err) << std::endl;
//         return -1;
//     }

//     const int BlockperGrid = (N + ThreadperBlock - 1) / ThreadperBlock;

//     err = cudaMemcpy(d_data, h_data, N * sizeof(Datatype), cudaMemcpyHostToDevice);
//     if (err != cudaSuccess) {
//         std::cerr << "CUDA memcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
//         return -1;
//     }

//     err = cudaMemset(d_sum, 0, sizeof(Datatype));
//     if (err != cudaSuccess) {
//         std::cerr << "CUDA memset d_sum failed: " << cudaGetErrorString(err) << std::endl;
//         return -1;
//     }

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);

//     // reduce_baseline<<<BlockperGrid, ThreadperBlock>>>(d_data, d_sum, N);
//     // reduce_shared<<<BlockperGrid, ThreadperBlock>>>(d_data, d_sum, N);
//     reduce_Parallelism<<<BlockperGrid, ThreadperBlock>>>(d_data, d_sum, N);

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float milliseconds = 0.0f;
//     cudaEventElapsedTime(&milliseconds, start, stop);

//     err = cudaMemcpy(&h_sum, d_sum, sizeof(Datatype), cudaMemcpyDeviceToHost);
//     if (err != cudaSuccess) {
//         std::cerr << "CUDA memcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
//         return -1;
//     }

//     bool is_right = CheckResult(h_sum, sum_cpu);
//     if (is_right) {
//         printf("the ans is right\n");
//     } else {
//         printf("the ans is wrong\n");
//     }

//     std::cout << "reduce_baseline latency = " << milliseconds << "ms" << std::endl;
//     std::cout << "CPU sum = " << sum_cpu << std::endl;
//     std::cout << "GPU sum = " << h_sum << std::endl;
//     std::cout << "CPU - GPU = " << sum_cpu - h_sum << std::endl;

//     // 释放内存
//     cudaFree(d_data);
//     cudaFree(d_sum);

//     // 释放 CUDA 事件
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     free(h_data);

//     return 0;
// }
