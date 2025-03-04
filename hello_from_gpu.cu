#include<iostream>

__global__ void hello_from_gpu()    // kernel code
{
  printf("Hello from GPU!\n");     // this is what we need to print  , this is on gpu
}

int main()                         // entry point
{
  hello_from_gpu<<<1, 1>>>();      //  threads per block, blocks per grid

  cudaDeviceSynchronize();         // wait for all the threads to complete their work 

  return 0;
}

// to compile the program   nvcc hello_from_gpu.cu -o hello_from_gpu.exe               .\hello_from_gpu.exe
