#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "graph.hpp"

class OpenCLSolution {
public:
    OpenCLSolution();
    ~OpenCLSolution();

    ///
    /// Run Dijkstra's shortest path on the GraphData provided to this function.
    ///
    void run_Dijkstra(GraphData *graph, int *source_vertices,
                      float *out_result_costs, int num_results);

private:
    ///
    /// Load and build an OpenCL program from source file
    /// \param gpu_context GPU context on which to load and build the program
    /// \param file_name   File name of source file that holds the kernels
    /// \return            Handle to the program
    ///
    cl_program load_and_build_program(cl_context gpu_context, const char *file_name);

    ///
    /// Gets the id of the first device from the context (from the NVIDIA SDK)
    ///
    cl_device_id get_first_dev(cl_context cx_GPU_context);

    ///
    /// Allocate memory for input CUDA buffers and copy the data into device memory
    ///
    void allocate_OCL_buffers(cl_context gpu_context, cl_command_queue command_queue,
                            GraphData *graph, cl_mem *vertex_array_device, cl_mem *edge_array_device,
                            cl_mem *weight_array_device, cl_mem *mask_array_device,
                            cl_mem *cost_array_device, cl_mem *updating_cost_array_device,
                            size_t global_work_size);

    ///
    /// Initialize OpenCL buffers for single run of Dijkstra
    ///
    void initialize_OCL_buffers(cl_command_queue command_queue, cl_kernel initialize_kernel,
                                GraphData *graph, size_t max_work_group_size);

    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    cl_platform_id platform;
    cl_context gpu_context;
    cl_int err_num;
};
