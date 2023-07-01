#pragma once

#include <iostream>
#define CL_TARGET_OPENCL_VERSION 100
#include <CL/cl.h>

// Cleanup any created OpenCL resources
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3]) {
    for (int i = 0; i < 3; ++i) {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);
}
