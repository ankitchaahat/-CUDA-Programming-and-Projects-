#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<chrono>

__global__ void add_gpu(int *a, int *b, int *c, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < n)
    {
        c[index] = b[index] + a[index];
    }
}

void add_cpu(int *a, int *b, int *c, int n)
{
    for(int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];

    }
}

int main()
{
    int n = 1 << 20; // 2^20 = 1048576
    int size = n * sizeof(int);

    int *h_a, int *h_b, int *h_c;
    int *d_a, int *d_b, int *d_c;
    
    h_a = (int *)malloc(size);
    h_b = (int *)malloc(size);
    h_c = (int *)malloc(size);

    // initialize arrays
    for(int i = 0; i < n; i++)
    {
        h_a[i] = i;
        h_b[i] = i*2;
    }

    // CPU timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    add_cpu(h_a, h_b, h_c, n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;

    // allocate device memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // GPU timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    add_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
    
     // Copy result back to host
     cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // --- Timing Output ---
    printf("CPU time: %.4f seconds\n", cpu_duration.count());
    printf("GPU time: %.4f milliseconds\n", milliseconds);
    printf("Speedup: %.2fx\n", (cpu_duration.count() * 1000) / milliseconds);

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;


}
