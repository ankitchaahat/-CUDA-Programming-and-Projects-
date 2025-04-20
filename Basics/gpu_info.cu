#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found.\n";
        return 1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "=== Device " << i << " ===\n";
        std::cout << "Name: " << prop.name << "\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Multiprocessors (SMs): " << prop.multiProcessorCount << "\n";
        std::cout << "Total Global Memory: " << (prop.totalGlobalMem >> 20) << " MB\n";
        std::cout << "Shared Memory per Block: " << (prop.sharedMemPerBlock >> 10) << " KB\n";
        std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "Warp Size: " << prop.warpSize << "\n";

        std::cout << "Max Grid Size: [" << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]\n";
        std::cout << "Max Block Dimensions: [" << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]\n";

        std::cout << "----------------------------------------\n";
    }

    return 0;
}
