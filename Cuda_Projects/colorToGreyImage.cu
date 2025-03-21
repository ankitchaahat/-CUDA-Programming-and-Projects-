#include <stdio.h>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"


// CUDA Kernel to convert RGB to Grayscale
__global__ void rgb_to_grayscale(unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        unsigned char r = d_input[idx];
        unsigned char g = d_input[idx + 1];
        unsigned char b = d_input[idx + 2];
        
        // Grayscale formula: 0.299*R + 0.587*G + 0.114*B
        unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        
        d_output[y * width + x] = gray;
    }
}

int main() {
    // Load image using stb_image
    int width, height, channels;
    unsigned char* h_input = stbi_load("D:/2023 all FILES/IMG_20210309_124347.jpg", &width, &height, &channels, 3);  // Force 3 channels (RGB)
    if (!h_input) {
        printf("Error: Failed to load image!\n");
        return -1;
    }

    size_t img_size = width * height * channels;
    size_t gray_size = width * height;

    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_output, gray_size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    rgb_to_grayscale<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    
    // Copy result back to host
    unsigned char* h_output = (unsigned char*)malloc(gray_size);
    cudaMemcpy(h_output, d_output, gray_size, cudaMemcpyDeviceToHost);

    // Save grayscale image using stb_image_write
    stbi_write_jpg("output.jpg", width, height, 1, h_output, 100);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);
    stbi_image_free(h_input);

    printf("Grayscale conversion completed. Check output.jpg\n");
    return 0;
}


// you have to save these two files in the local folder   stb_image.h and stb_image_write.h  save them from official github repo
