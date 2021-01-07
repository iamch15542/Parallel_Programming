#ifndef __HOSTFE__
#define __HOSTFE__

extern "C"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage);

#endif