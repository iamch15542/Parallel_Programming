#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float stepX, float stepY, float lowerX, float lowerY, int* img_result, int maxIterations, int pitch) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    
    float x = lowerX + (blockIdx.x * blockDim.x + threadIdx.x) * stepX;
    float y = lowerY + (blockIdx.y * blockDim.y + threadIdx.y) * stepY;
    
    float z_re = x, z_im = y;
    int i;
    for (i = 0; i < maxIterations; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
    }
    int* idx = (int*)((char*)img_result + (blockIdx.y * blockDim.y + threadIdx.y) * pitch) + (blockIdx.x * blockDim.x + threadIdx.x);
    *idx = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // Declare the host memory
    int *h_result;
    cudaHostAlloc((void **)&h_result, resX * resY * sizeof(int), cudaHostAllocDefault);

    // Declare the cuda memory
    int *c_result;
    size_t pitch;
    cudaMallocPitch((void **)&c_result, &pitch, sizeof(int) * resX, resY);

    dim3 blockSize(16, 16);
    dim3 numBlock(resX / 16, resY / 16);

    mandelKernel<<<numBlock, blockSize>>>(stepX, stepY, lowerX, lowerY, c_result, maxIterations, pitch);

    // 等待 GPU 所有 thread 完成
    cudaDeviceSynchronize();

    // 將 Device 的資料傳回給 Host
    cudaMemcpy2D(h_result, resX * sizeof(int), c_result, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);

    for(int i = 0; i < resX * resY; ++i) {
        *(img+i) = *(h_result+i);
    }

    // free memory
    cudaFreeHost(h_result);
    cudaFree(c_result);
}
