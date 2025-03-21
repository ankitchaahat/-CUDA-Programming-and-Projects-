#include<stdio.h>
#include<cuda_runtime.h>

__global__ void finding_divisor(int *input, int *output, int n) {
    __shared__ int shared_memory[256];  

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        shared_memory[tid] = input[idx];  
    }
    __syncthreads();

    if (idx < n) {
        int num = shared_memory[tid];  
        int count = 0;  

        for (int i = 1; i <= num; i++) {
            if (num % i == 0) {
                output[idx * 10 + count] = i;  
                count++;
            }
        }
        output[idx * 10 + count] = -1; // End marker
    }
}

int main() {
    int n = 256;
    int *h_input, *h_output;
    int *d_input, *d_output;

    h_input = (int *)malloc(n * sizeof(int));
    h_output = (int *)malloc(n * 10 * sizeof(int));  // Max 10 divisors per number

    for (int i = 0; i < n; i++) {
        h_input[i] = i + 1;
    }

    cudaMalloc((void **)&d_input, n * sizeof(int));
    cudaMalloc((void **)&d_output, n * 10 * sizeof(int));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int numBlocks = (n + threads_per_block - 1) / threads_per_block;

    finding_divisor<<<numBlocks, threads_per_block>>>(d_input, d_output, n);

    cudaMemcpy(h_output, d_output, n * 10 * sizeof(int), cudaMemcpyDeviceToHost);

    // Print divisors
    for (int i = 0; i < n; i++) {
        printf("Divisors of %d: ", h_input[i]);
        for (int j = 0; j < 10; j++) {
            if (h_output[i * 10 + j] == -1) break;
            printf("%d ", h_output[i * 10 + j]);
        }
        printf("\n");
    }

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

