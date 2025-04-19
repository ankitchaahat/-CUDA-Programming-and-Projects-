#include<cuda_runtime.h>
#include<iostream>
#include<chrono>

#define N (1 << 20) // 2^20 = 1048576
#define NUM_STREAMS 4


__global__ void intArray(int *arr, int value, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size){
        arr[idx] = value;
    }
}

int main()
{
    int *h_data = new int[N];
    int *d_data;

    cudaMalloc((void **)&d_data, N*sizeof(int));

    // stream creation
    cudaStream_t streams[NUM_STREAMS];
    for(int i = 0; i < NUM_STREAMS; i++){
        cudaStreamCreate(&streams[i]);
    }

    int chunksize = N / NUM_STREAMS;
    int blocksize = 256;
    int gridsize = (chunksize + blocksize -1 ) / blocksize;

    // launch kernel in parallel stremas
    for(int i = 0; i < NUM_STREAMS; i++){
        int offset = i * chunksize;
        intArray<<<gridsize, blocksize, 0, streams[i]>>>(d_data + offset, i+1, chunksize);
    }

    // copy data from device to host
    cudaMemcpy(h_data, d_data, N*sizeof(int), cudaMemcpyDeviceToHost);

    // wait for all streams to finish
    for(int i = 0; i < NUM_STREAMS; i++){
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // validate results
    bool correct = true;
    for(int i = 0; i < N; i++){
        if(h_data[i] != (i / chunksize) + 1){
            correct = false;
            break;
        }
    }
    if(correct){
        std::cout << "Data is correct!" << std::endl;
    } else {
        std::cout << "Data is incorrect!" << std::endl;
    }

    // print first 10 elements of the result
    std::cout << "First 10 elements of the result: " << std::endl;
    for(int i = 0; i < 10; i++){
        std::cout << h_data[i] << " ";
    }

    // free memory
    delete[] h_data;
    cudaFree(d_data);
    return 0;

    
}
    
