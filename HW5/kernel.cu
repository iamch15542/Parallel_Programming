#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float stepX, float stepY, float lowerX, float lowerY, int* img_result, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int idx = (blockIdx.y * blockDim.y + threadIdx.y) * (gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    float x = lowerX + (blockIdx.x * blockDim.x + threadIdx.x) * stepX;
    float y = lowerY + (blockIdx.y * blockDim.y + threadIdx.y) * stepY;
    
    float z_re = x, z_im = y, re, im, tmp;
    int i;
    for (i = 0; i < maxIterations; ++i) {
        re = __fmul_rn(z_re, z_re);
        im = __fmul_rn(z_im, z_im);
        tmp = __fadd_rn(re, im);
        if (tmp > 4.f)
            break;

        float new_re = __fsub_rn(re, im);
        float new_im = 2.f * __fmul_rn(z_re, z_im);
        z_re = __fadd_rn(x, new_re);
        z_im = __fadd_rn(y, new_im);
    }
    img_result[idx] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // Declare the host memory
    // int *h_result = (int *)malloc(resX * resY * sizeof(int));

    // Declare the cuda memory
    int *c_result;
    cudaMalloc(&c_result, resX * resY * sizeof(int));

    dim3 blockSize(16, 16);
    dim3 numBlock(resX / 16, resY / 16);

    mandelKernel<<<numBlock, blockSize>>>(stepX, stepY, lowerX, lowerY, c_result, maxIterations);

    // 將 Device 的資料傳回給 Host
    cudaMemcpy(img, c_result, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(c_result);
}
