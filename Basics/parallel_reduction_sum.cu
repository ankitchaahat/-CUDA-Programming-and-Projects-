// Sum of array elements using parallel reduction

#include <iostream>
#include<cuda_runtime.h>
#define N 256

__global__ void parallel_reduction(float *input, float *output,  int n)
{
    __shared__ float shared_memory[N];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < N)
    {
        shared_memory[tid] = input[idx];
    }
    __syncthreads();

    // parallel reduction
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if(tid < stride)
        {
            shared_memory[tid] += shared_memory[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        output[blockIdx.x] = shared_memory[0];  // saving all the sum of each block in the output array
    }
}

int main()
{
    int n = N;
    float *h_input, *h_output;
    float *d_input, *d_output;
    float sum = 0;

    h_input = (float *)malloc(n * sizeof(float));
    h_output = (float *)malloc(n * sizeof(float));

    for(int i = 0; i < n; i++)
    {
        h_input[i] = i;
    }

    cudaMalloc((void **)&d_input, n * sizeof(float));
    cudaMalloc((void **)&d_output, n * sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float),  cudaMemcpyHostToDevice);

    int threads_per_block = n;
    int numBlocks = (N + threads_per_block - 1) / threads_per_block;

    parallel_reduction<<<numBlocks, threads_per_block>>>(d_input, d_output, n);

    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++)
    {
        sum += h_output[i];
    }

    std::cout << "Sum of array elements: " << sum << std::endl;

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
