#include <iostream>
#include <fstream>
#include <sstream>

#include "opencl_sol.hpp"
#include "helper.hpp"
#include "constants.hpp"
#include "error.hpp"

OpenCLSolution::OpenCLSolution() {
    // First, select an OpenCL platform to run on. For this example, we
    // simply choose the first available platform. Normally, you would
    // query for all available platform an select the most appropriate one.
    cl_uint num_platforms;
    err_num = clGetPlatformIDs(1, &platform, &num_platforms);
    std::cout << "Number of OpenCL Platforms: " << num_platforms << std::endl;
    if (err_num != CL_SUCCESS || num_platforms <= 0) {
        std::cout << "Failed to find any OpenCL platforms." << std::endl;
        exit(1);
    }

    cl_context_properties context_properties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    // Create the OpenCL context on available GPU devices
    gpu_context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &err_num);
    if (err_num != CL_SUCCESS)
        std::cout << "No GPU devices found: " << err_num << std::endl;
}

OpenCLSolution::~OpenCLSolution() {
    clReleaseContext(gpu_context);
}

void OpenCLSolution::run_Dijkstra(GraphData *graph, int *source_vertices,
                                  float *out_result_costs, int num_results) {
    cl_device_id device_id = get_max_flops_dev(gpu_context);

    // Create command queue
    cl_int err_num;
    cl_command_queue command_queue;
    command_queue = clCreateCommandQueueWithProperties(gpu_context, device_id, 0, &err_num);
    check_error(err_num, CL_SUCCESS);

    // Program handle
    cl_program program = load_and_build_program(gpu_context, "include/kernel.cl");
    if (program == NULL) return;

    // Get the max workgroup size
    size_t max_work_group_size;
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    check_error(err_num, CL_SUCCESS);
    std::cout << "MAX_WORKGROUP_SIZE: " << max_work_group_size << std::endl;
    std::cout << "Computing '" << num_results << "' results." << std::endl;

    // Set # of work items in work group and total in 1 dimensional range
    size_t local_work_size = max_work_group_size;
    size_t global_work_size = round_work_size_up(local_work_size, graph->vertex_count);

    cl_mem vertex_array_device;
    cl_mem edge_array_device;
    cl_mem weight_array_device;
    cl_mem mask_array_device;
    cl_mem cost_array_device;
    cl_mem updating_cost_array_device;

    // Allocate buffers in Device memory
    allocate_OCL_buffers(gpu_context, command_queue, graph, &vertex_array_device,
                         &edge_array_device, &weight_array_device, &mask_array_device,
                         &cost_array_device, &updating_cost_array_device, global_work_size);

    // Create the Kernels
    cl_kernel initialize_buffers_kernel;
    initialize_buffers_kernel = clCreateKernel(program, "initializeBuffers", &err_num);
    check_error(err_num, CL_SUCCESS);

    // Set the args values and check for errors
    err_num |= clSetKernelArg(initialize_buffers_kernel, 0, sizeof(cl_mem), &mask_array_device);
    err_num |= clSetKernelArg(initialize_buffers_kernel, 1, sizeof(cl_mem), &cost_array_device);
    err_num |= clSetKernelArg(initialize_buffers_kernel, 2, sizeof(cl_mem), &updating_cost_array_device);
    // 3rd arg set below in loop
    err_num |= clSetKernelArg(initialize_buffers_kernel, 4, sizeof(int), &graph->vertex_count);
    check_error(err_num, CL_SUCCESS);

    // Kernel 1
    cl_kernel ssspKernel1;
    ssspKernel1 = clCreateKernel(program, "DijkstraKernelPhase1", &err_num);
    check_error(err_num, CL_SUCCESS);
    err_num |= clSetKernelArg(ssspKernel1, 0, sizeof(cl_mem), &vertex_array_device);
    err_num |= clSetKernelArg(ssspKernel1, 1, sizeof(cl_mem), &edge_array_device);
    err_num |= clSetKernelArg(ssspKernel1, 2, sizeof(cl_mem), &weight_array_device);
    err_num |= clSetKernelArg(ssspKernel1, 3, sizeof(cl_mem), &mask_array_device);
    err_num |= clSetKernelArg(ssspKernel1, 4, sizeof(cl_mem), &cost_array_device);
    err_num |= clSetKernelArg(ssspKernel1, 5, sizeof(cl_mem), &updating_cost_array_device);
    err_num |= clSetKernelArg(ssspKernel1, 6, sizeof(int), &graph->vertex_count);
    err_num |= clSetKernelArg(ssspKernel1, 7, sizeof(int), &graph->edge_count);
    check_error(err_num, CL_SUCCESS);

    // Kernel 2
    cl_kernel ssspKernel2;
    ssspKernel2 = clCreateKernel(program, "DijkstraKernelPhase2", &err_num);
    check_error(err_num, CL_SUCCESS);
    err_num |= clSetKernelArg(ssspKernel2, 0, sizeof(cl_mem), &vertex_array_device);
    err_num |= clSetKernelArg(ssspKernel2, 1, sizeof(cl_mem), &edge_array_device);
    err_num |= clSetKernelArg(ssspKernel2, 2, sizeof(cl_mem), &weight_array_device);
    err_num |= clSetKernelArg(ssspKernel2, 3, sizeof(cl_mem), &mask_array_device);
    err_num |= clSetKernelArg(ssspKernel2, 4, sizeof(cl_mem), &cost_array_device);
    err_num |= clSetKernelArg(ssspKernel2, 5, sizeof(cl_mem), &updating_cost_array_device);
    err_num |= clSetKernelArg(ssspKernel2, 6, sizeof(int), &graph->vertex_count);

    check_error(err_num, CL_SUCCESS);

    int *mask_array_host = (int*) malloc(sizeof(int) * graph->vertex_count);

    for (int i = 0; i < num_results; i++ ) {
        err_num |= clSetKernelArg(initialize_buffers_kernel, 3, sizeof(int), &source_vertices[i]);
        check_error(err_num, CL_SUCCESS);

        // Initialize mask array to false, C and U to infiniti
        initialize_OCL_buffers(command_queue, initialize_buffers_kernel, graph, max_work_group_size);

        // Read mask array from device -> host
        cl_event read_done;
        err_num = clEnqueueReadBuffer(command_queue, mask_array_device, CL_FALSE, 0, sizeof(int) * graph->vertex_count,
                                      mask_array_host, 0, NULL, &read_done);
        check_error(err_num, CL_SUCCESS);
        clWaitForEvents(1, &read_done);

        while(!mask_array_empty(mask_array_host, graph->vertex_count))
        {

            // In order to improve performance, we run some number of iterations
            // without reading the results.  This might result in running more iterations
            // than necessary at times, but it will in most cases be faster because
            // we are doing less stalling of the GPU waiting for results.
            for(int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; asyncIter++)
            {
                size_t local_work_size = max_work_group_size;
                size_t global_work_size = round_work_size_up(local_work_size, graph->vertex_count);

                // execute the kernel
                err_num = clEnqueueNDRangeKernel(command_queue, ssspKernel1, 1, 0, &global_work_size, &local_work_size,
                                                 0, NULL, NULL);
                check_error(err_num, CL_SUCCESS);

                err_num = clEnqueueNDRangeKernel(command_queue, ssspKernel2, 1, 0, &global_work_size, &local_work_size,
                                                 0, NULL, NULL);
                check_error(err_num, CL_SUCCESS);
            }
            err_num = clEnqueueReadBuffer(command_queue, mask_array_device, CL_FALSE, 0, sizeof(int) * graph->vertex_count,
                                          mask_array_host, 0, NULL, &read_done);
            check_error(err_num, CL_SUCCESS);
            clWaitForEvents(1, &read_done);
        }

        // Copy the result back
        err_num = clEnqueueReadBuffer(command_queue, cost_array_device, CL_FALSE, 0, sizeof(float) * graph->vertex_count,
                                      &out_result_costs[i * graph->vertex_count], 0, NULL, &read_done);
        check_error(err_num, CL_SUCCESS);
        clWaitForEvents(1, &read_done);
    }

    free(mask_array_host);

    clReleaseMemObject(vertex_array_device);
    clReleaseMemObject(edge_array_device);
    clReleaseMemObject(weight_array_device);
    clReleaseMemObject(mask_array_device);
    clReleaseMemObject(cost_array_device);
    clReleaseMemObject(updating_cost_array_device);

    clReleaseKernel(initialize_buffers_kernel);
    clReleaseKernel(ssspKernel1);
    clReleaseKernel(ssspKernel2);

    clReleaseCommandQueue(command_queue);
    clReleaseProgram(program);
    std::cout << "Computed '" << num_results << "' results" << std::endl;
}

cl_program OpenCLSolution::load_and_build_program(cl_context gpu_context, const char *file_name) {
    pthread_mutex_lock(&mutex);

    cl_int err_num;
    cl_program program;

    // Load the OpenCL source code from the .cl file
    std::ifstream kernel_file(file_name, std::ios::in);
    if (!kernel_file.is_open()) {
        std::cerr << "Failed to open file for reading: " << file_name << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernel_file.rdbuf();

    std::string src_std_str = oss.str();
    const char *source = src_std_str.c_str();

    check_error(source != NULL, true);

    // Create the program for all GPUs in the context
    program = clCreateProgramWithSource(gpu_context, 1, static_cast<const char **>(&source), NULL, &err_num);
    check_error(err_num, CL_SUCCESS);
    // build the program for all devices on the context
    err_num = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err_num != CL_SUCCESS) {
        char cBuildLog[10240];
        clGetProgramBuildInfo(program, get_first_dev(gpu_context), CL_PROGRAM_BUILD_LOG,
                              sizeof(cBuildLog), cBuildLog, NULL);
        std::cerr << cBuildLog << std::endl;
        check_error(err_num, CL_SUCCESS);
    }

    pthread_mutex_unlock(&mutex);
    return program;
}

cl_device_id OpenCLSolution::get_first_dev(cl_context cx_GPU_context) {
                                           size_t sz_parm_data_bytes;
                                           cl_device_id *cd_devices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cx_GPU_context, CL_CONTEXT_DEVICES, 0, NULL, &sz_parm_data_bytes);
    cd_devices = static_cast<cl_device_id *>(malloc(sz_parm_data_bytes));

    clGetContextInfo(cx_GPU_context, CL_CONTEXT_DEVICES, sz_parm_data_bytes, cd_devices, NULL);

    cl_device_id first = cd_devices[0];
    free(cd_devices);

    return first;
}

void OpenCLSolution::allocate_OCL_buffers(cl_context gpu_context, cl_command_queue command_queue,
                            GraphData *graph, cl_mem *vertex_array_device, cl_mem *edge_array_device,
                            cl_mem *weight_array_device, cl_mem *mask_array_device,
                            cl_mem *cost_array_device, cl_mem *updating_cost_array_device,
                            size_t global_work_size) {
    cl_int err_num;
    cl_mem host_vertex_array_buffer;
    cl_mem host_edge_array_buffer;
    cl_mem host_weight_array_buffer;

    // First, need to create OpenCL Host buffers that can be copied to device buffers
    host_vertex_array_buffer = clCreateBuffer(gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                        sizeof(int) * graph->vertex_count, graph->vertex_array, &err_num);
    check_error(err_num, CL_SUCCESS);

    host_edge_array_buffer = clCreateBuffer(gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                        sizeof(int) * graph->edge_count, graph->edge_array, &err_num);
    check_error(err_num, CL_SUCCESS);

    host_weight_array_buffer = clCreateBuffer(gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                        sizeof(float) * graph->edge_count, graph->weight_array, &err_num);
    check_error(err_num, CL_SUCCESS);

    // Now create all of the GPU buffers
    *vertex_array_device = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, sizeof(int) * global_work_size, NULL, &err_num);
    check_error(err_num, CL_SUCCESS);
    *edge_array_device = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, sizeof(int) * graph->edge_count, NULL, &err_num);
    check_error(err_num, CL_SUCCESS);
    *weight_array_device = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, sizeof(float) * graph->edge_count, NULL, &err_num);
    check_error(err_num, CL_SUCCESS);
    *mask_array_device = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(int) * global_work_size, NULL, &err_num);
    check_error(err_num, CL_SUCCESS);
    *cost_array_device = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(float) * global_work_size, NULL, &err_num);
    check_error(err_num, CL_SUCCESS);
    *updating_cost_array_device = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(float) * global_work_size, NULL, &err_num);
    check_error(err_num, CL_SUCCESS);

    // Now queue up the data to be copied to the device
    err_num = clEnqueueCopyBuffer(command_queue, host_vertex_array_buffer, *vertex_array_device, 0, 0,
                                sizeof(int) * graph->vertex_count, 0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);

    err_num = clEnqueueCopyBuffer(command_queue, host_edge_array_buffer, *edge_array_device, 0, 0,
                                sizeof(int) * graph->edge_count, 0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);

    err_num = clEnqueueCopyBuffer(command_queue, host_weight_array_buffer, *weight_array_device, 0, 0,
                                sizeof(float) * graph->edge_count, 0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);

    clReleaseMemObject(host_vertex_array_buffer);
    clReleaseMemObject(host_edge_array_buffer);
    clReleaseMemObject(host_weight_array_buffer);
}

void OpenCLSolution::initialize_OCL_buffers(cl_command_queue command_queue, cl_kernel initialize_kernel,
                                            GraphData *graph, size_t max_work_group_size) {
    cl_int err_num;
    // Set # of work items in work group and total in 1 dimensional range
    size_t local_work_size = max_work_group_size;
    size_t global_work_size = round_work_size_up(local_work_size, graph->vertex_count);

    err_num = clEnqueueNDRangeKernel(command_queue, initialize_kernel, 1, NULL,
                                     &global_work_size, &local_work_size, 0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
}
