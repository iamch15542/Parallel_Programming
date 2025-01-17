__kernel void convolution(const __global float *inputImage, __constant float *filter, __global float *outputImage,
                          const int filterWidth, const int imageHeight, const int imageWidth) 
{
    const int i = get_global_id(0) / imageWidth; // col
    const int j = get_global_id(0) % imageWidth; // row

    int sum = 0;
    int halfwidth = filterWidth / 2;
    // Apply the filter to the neighborhood
    for(int fi = -halfwidth; fi <= halfwidth; ++fi) {
        for(int fj = -halfwidth; fj <= halfwidth; ++fj) {
            if(i + fi >= 0 && i + fi < imageHeight && j + fj >= 0 && j + fj < imageWidth && filter[(fi + halfwidth) * filterWidth + fj + halfwidth]) {
                sum += inputImage[(i + fi) * imageWidth + j + fj] * filter[(fi + halfwidth) * filterWidth + fj + halfwidth];
            }
        }
    }
    outputImage[i * imageWidth + j] = sum;
}