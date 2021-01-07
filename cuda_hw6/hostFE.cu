#include "hostFE.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void convolution(float *inputImage, float *filter, float *outputImage, int filterWidth, int imageHeight, int imageWidth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // col
    int j = blockIdx.y * blockDim.y + threadIdx.y; // row

    int sum = 0;
    int halfwidth = filterWidth / 2;
    // Apply the filter to the neighborhood
    for(int fi = -halfwidth; fi <= halfwidth; ++fi) {
        for(int fj = -halfwidth; fj <= halfwidth; ++fj) {
            if(i + fi >= 0 && i + fi < imageHeight && j + fj >= 0 && j + fj < imageWidth) {
                sum += inputImage[(i + fi) * imageWidth + j + fj] * filter[(fi + halfwidth) * filterWidth + fj + halfwidth];
            }
        }
    }
    outputImage[i * imageWidth + j] = sum;
}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage)
{
    // Declare the cuda memory
    int data_size = imageHeight * imageWidth * sizeof(float);

    float *d_inputImage;
    size_t pitch;
    cudaMallocPitch((void **)&d_inputImage, &pitch, imageWidth * sizeof(float), imageHeight);

    float *d_filter;
    size_t pitch2;
    cudaMallocPitch((void **)&d_filter, &pitch2, filterWidth * sizeof(float), filterWidth);

    float *img_result;
    size_t pitch3;
    cudaMallocPitch((void **)&img_result, &pitch3, imageWidth * sizeof(float), imageHeight);
    printf("good\n")
    // copy data
    // cudaMemcpy(d_inputImage, inputImage, data_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_filter, filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_inputImage, imageWidth * sizeof(float), inputImage, pitch, imageWidth * sizeof(float), imageHeight, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_filter, filterWidth * sizeof(float), filter, pitch2, filterWidth * sizeof(float), filterWidth, cudaMemcpyHostToDevice);

    dim3 blockSize(20, 20);
    dim3 numBlock(imageHeight / 20, imageWidth / 20);

    convolution<<<numBlock, blockSize>>>(d_inputImage, d_filter, img_result, filterWidth, imageHeight, imageWidth);

    // 等待 GPU 所有 thread 完成
    cudaDeviceSynchronize();

    // 將 Device 的資料傳回給 Host
    cudaMemcpy(outputImage, img_result, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(outputImage, imageWidth * sizeof(float), img_result, pitch3, imageWidth * sizeof(float), imageHeight, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_inputImage);
    cudaFree(d_filter);
    cudaFree(img_result);
}