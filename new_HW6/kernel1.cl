__kernel void convolution(const __global float *inputImage, __constant float *filter, __global float *outputImage,
                          const int filterWidth, const int imageHeight, const int imageWidth) 
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int fi, fj;

    int halfwidth = filterWidth / 2;
    int sum = 0;

    int th_i = get_local_id(0) * 2;
    int th_j = get_local_id(1);
    int in_i = i * 2 - halfwidth;
    __local int localImage[256];

    if(th_j == 0) {
        #pragma unroll 16
        for(int tmp_j = 0; tmp_j < 16; ++tmp_j) {
            if(in_i >= 0 && j - halfwidth + tmp_j >= 0 && in_i < imageHeight && j - halfwidth + tmp_j < imageWidth) {
                localImage[th_i * 16 + tmp_j] = inputImage[in_i * imageWidth + j - halfwidth + tmp_j];
            } else {
                localImage[th_i * 16 + tmp_j] = 0;
            }

            if(in_i + 1 >= 0 && j - halfwidth + tmp_j >= 0 && in_i + 1 < imageHeight && j - halfwidth + tmp_j < imageWidth) {
                localImage[(th_i + 1) * 16 + tmp_j] = inputImage[(in_i + 1) * imageWidth + j - halfwidth + tmp_j];
            } else {
                localImage[(th_i + 1) * 16 + tmp_j] = 0;
            }
        }
    }


    // if(th_i == 0 && th_j == 0) {
    //     #pragma unroll 16
    //     for(int tmp_i = 0; tmp_i < 16; ++tmp_i) {
    //         #pragma unroll 16
    //         for(int tmp_j = 0; tmp_j < 16; ++tmp_j) {
    //             if(i - halfwidth + tmp_i >= 0 && j - halfwidth + tmp_j >= 0 && i - halfwidth + tmp_i < imageHeight && j - halfwidth + tmp_j < imageWidth) {
    //                 localImage[tmp_i * 16 + tmp_j] = inputImage[(i - halfwidth + tmp_i) * imageWidth + j - halfwidth + tmp_j];
    //             } else {
    //                 localImage[tmp_i * 16 + tmp_j] = 0;
    //             }
    //         }
    //     }
    // }

    // localImage[th_i * 8 + th_j]           = inputImage[in_i * imageWidth + in_j];
    // localImage[th_i * 8 + th_j + 1]       = inputImage[in_i * imageWidth + in_j + 1];
    // localImage[(th_i + 1) * 8 + th_j]     = inputImage[(in_1 + 1) * imageWidth + in_j];
    // localImage[(th_i + 1) * 8 + th_j + 1] = inputImage[(in_i + 1) * imageWidth + in_j + 1];

    barrier(CLK_LOCAL_MEM_FENCE);

    if(i == 33 && j == 33) {
        for(int tt = 0; tt < 16; ++tt) {
            for(int kk = 0; kk < 16; ++kk) {
                printf("%d ", localImage[tt * 16 + kk]);
            }
            printf("\n");
        }
    }

    // Apply the filter to the neighborhood
    for(fi = -halfwidth; fi <= halfwidth; ++fi) {
        int local_x = th_i + halfwidth;
        for(fj = -halfwidth; fj <= halfwidth; ++fj) {
            int local_y = th_j + halfwidth;
            if(filter[(fi + halfwidth) * filterWidth + fj + halfwidth]) {
                sum += localImage[(local_x + fi) * 16 + local_y + fj] * filter[(fi + halfwidth) * filterWidth + fj + halfwidth];
            }
        }
    }
    outputImage[i * imageWidth + j] = sum;
}