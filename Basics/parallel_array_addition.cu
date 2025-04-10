#include<stdio.h>
#include<cuda_runtime.h>


__global__ void add(int *a, int *b, int *c, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < n)
    {
        c[index] = a[index] + b[index];
    }
    
}
int main()
{
    int n = 1024;

    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    size_t  size = n * sizeof(int);

    // Allocate host memory
    h_a = (int *)malloc(size);
    h_b = (int *)malloc(size);
    h_c = (int *)malloc(size);

    // initialize input vectors
    for(int i = 0; i < n; i++)
    {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // copy vectors from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // launch kernel
    int threadsPerBlock(256);
    int blocksPerGrid((n + threadsPerBlock - 1) / threadsPerBlock);
    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // copy result from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // print result
    for(int i = 0 ; i < n; i++)
    {
        printf(" %d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // free host memory
    free(h_a);
    free(h_b);
    free(h_c);
}
