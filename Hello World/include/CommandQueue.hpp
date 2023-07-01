#pragma once

#include <iostream>
#define CL_TARGET_OPENCL_VERSION 100
#include <CL/cl.h>

cl_command_queue CreateCommandQueue(cl_context context,
                                    cl_device_id *device) {
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Failed call to clGetContextInfo(..., GL_CONTEXT_DEVICES, ...)";
        return NULL;
    }

    if (deviceBufferSize <= 0) {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES,
                              deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.
    // In a real program, you would likely use all available
    // devices or choose the highest performance device based on
    // OpenCL device queries.

    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL) {
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}
