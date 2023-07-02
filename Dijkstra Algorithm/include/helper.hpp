#pragma once
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

///
/// Round the local work size up to the next multiple of the size
///
inline int round_work_size_up(int group_size, int global_size) {
    int remainder = global_size % group_size;
    if (remainder == 0) return global_size;
    else return global_size + group_size - remainder;
}

///
/// Check whether the mask array is empty. This tells the algorithm whether
/// it needs to continue running or not
///
inline bool mask_array_empty(int *mask_array, int count) {
    for (int i = 0; i < count; ++i) {
        if (mask_array[i] == 1) return false;
    }
    return true;
}


///
/// Gets the id of device with maximal FLOPS from the context (from NVIDIA SDK)
/// Best computing performance
///
static cl_device_id get_max_flops_dev(cl_context cx_GPU_context) {
    size_t sz_parm_data_bytes;
    cl_device_id *cd_devices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cx_GPU_context, CL_CONTEXT_DEVICES, 0, NULL, &sz_parm_data_bytes);
    cd_devices = static_cast<cl_device_id *>(malloc(sz_parm_data_bytes));
    size_t device_count = sz_parm_data_bytes / sizeof(cl_device_id);

    clGetContextInfo(cx_GPU_context, CL_CONTEXT_DEVICES, sz_parm_data_bytes, cd_devices, NULL);

    cl_device_id max_flops_device = cd_devices[0];
    int max_flops = 0;

    size_t current_device;

    // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_uint compute_units;
    // CL_DEVICE_MAX_CLOCK_FREQUENCY
    cl_uint clock_frequency;

    for (current_device = 0; current_device < device_count; ++current_device) {
        // CL_DEVICE_MAX_COMPUTE_UNITS
        cl_uint compute_units;
        clGetDeviceInfo(cd_devices[current_device], CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(compute_units), &compute_units, NULL);

        // CL_DEVICE_MAX_CLOCK_FREQUENCY
        cl_uint clock_frequency;
        clGetDeviceInfo(cd_devices[current_device],
                CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency),
                &clock_frequency, NULL);

        int flops = compute_units * clock_frequency;
        if (flops > max_flops) {
            max_flops = flops;
            max_flops_device = cd_devices[current_device];
        }
    }

    free(cd_devices);
    return max_flops_device;
}
