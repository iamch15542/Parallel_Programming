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
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_inputImage);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_filter);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_outputImage);
    
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&filterWidth);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&imageHeight);
    clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&imageWidth);
    
    printf("Set Kernel Arg Success\n");
    
    // execute kernel
    size_t localws[2] = {2, 2};
    size_t globalws[2] = {imageWidth, imageHeight};
    status = clEnqueueNDRangeKernel(imgqueue, kernel, 2, 0, globalws, localws, 0, NULL, NULL);
    CHECK(status, "clEnqueueNDRangeKernel");

    printf("Finish kernel\n");

    if(status == CL_SUCCESS) {
        status = clEnqueueReadBuffer(imgqueue, d_outputImage, CL_TRUE, 0, sizeof(cl_float) * imageSize, outputImage, NULL, NULL, NULL);
    }
    CHECK(status, "clEnqueueReadBuffer");

    // release opencl
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    // clReleaseMemObject(d_filterWidth);
    clReleaseMemObject(d_filter);
    // clReleaseMemObject(d_imageHeight);
    // clReleaseMemObject(d_imageWidth);
    clReleaseMemObject(d_inputImage);
    clReleaseMemObject(d_outputImage);
    clReleaseCommandQueue(imgqueue);
    clReleaseContext(context);
    return 0;
}