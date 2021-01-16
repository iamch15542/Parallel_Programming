__kernel void convolution(const __global float *inputImage, __constant float *filter, __global float *outputImage,
                          const int filterWidth, const int imageHeight, const int imageWidth) 
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int fi, fj;

    int halfwidth = filterWidth / 2;
    int sum = 0;
    // Apply the filter to the neighborhood
    for(fi = -halfwidth; fi <= halfwidth; ++fi) {
        for(fj = -halfwidth; fj <= halfwidth; ++fj) {
            if(filter[(fi + halfwidth) * filterWidth + fj + halfwidth]) {
                if(i + fi >= 0 && i + fi < imageHeight && j + fj >= 0 && j + fj < imageWidth) {
                    sum += inputImage[(i + fi) * imageWidth + j + fj] * filter[(fi + halfwidth) * filterWidth + fj + halfwidth];
                }
            }
        }
    }
    outputImage[i * imageWidth + j] = sum;
}