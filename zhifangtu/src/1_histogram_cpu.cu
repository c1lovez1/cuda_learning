#include "cuda_error_check.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <vector>

#define NUM_BINS 256
#define DATA_SIZE 1024

void histogram_cpu(std::vector<int>&data,std::vector<int>&hist){
    std::fill(hist.begin(),hist.end(),0);
    for (int value : data){
        hist[value]++;
    }
}


int main()
{
    std::vector<int> data(DATA_SIZE);
    std::vector<int> hist(NUM_BINS);

    for (int i = 0; i < DATA_SIZE; i++)
    {
        data[i] = rand() % NUM_BINS;
    }

    histogram_cpu(data,hist);

    // 打印直方图结果
    for (int i = 0; i < NUM_BINS; i++) {
        std::cout << "Bin " << i << ": " << hist[i] << std::endl;
    }

    return 0;
}
