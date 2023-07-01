#pragma once

#include <iostream>
#define CL_TARGET_OPENCL_VERSION 100
#include <CL/cl.h>

const int ARRAY_SIZE = 1000;

bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
                      float *a, float *b) {
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                   CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, a,
                                   NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                   CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, b,
                                   NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE,
                                   NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL ||
        memObjects[2] == NULL) {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}
