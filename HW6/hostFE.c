#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageHeight * imageWidth;

    // create commandqueue
    cl_command_queue imgqueue;
    imgqueue = clCreateCommandQueue(*context, *device, 0, &status);
    CHECK(status, "clCreateCommandQueue");

    // create buffer on device
    cl_mem d_inputImage = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * imageSize, inputImage, &status);
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * filterSize, filter, &status);
    cl_mem d_outputImage = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, sizeof(float) * imageSize, NULL, &status);
    
    // cl_mem d_filterWidth = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float), &filterWidth, &status);
    // cl_mem d_imageHeight = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &imageHeight, &status);
    // cl_mem d_imageWidth = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &imageWidth, &status);

    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    CHECK(status, "clCreateKernel");
    
    // set kernel Arg
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_inputImage);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_filter);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_outputImage);
    
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&filterWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void*)&imageHeight);
    clSetKernelArg(kernel, 5, sizeof(int), (void*)&imageWidth);
    
    // execute kernel
    // size_t localws[2] = {2, 2};
    size_t globalws = imageSize;
    status = clEnqueueNDRangeKernel(imgqueue, kernel, 1, 0, &globalws, 0, 0, NULL, NULL);
    CHECK(status, "clEnqueueNDRangeKernel");

    if(status == CL_SUCCESS) {
        status = clEnqueueReadBuffer(imgqueue, d_outputImage, CL_TRUE, 0, sizeof(float) * imageSize, outputImage, NULL, NULL, NULL);
    }
    CHECK(status, "clEnqueueReadBuffer");

    // release opencl
    clReleaseKernel(kernel);
    
    clReleaseMemObject(d_inputImage);
    clReleaseMemObject(d_filter);
    clReleaseMemObject(d_outputImage);

    // clReleaseMemObject(d_filterWidth);
    // clReleaseMemObject(d_imageHeight);
    // clReleaseMemObject(d_imageWidth);

    clReleaseCommandQueue(imgqueue);
    return 0;
}