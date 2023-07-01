#include <sstream>
#include <iostream>
#include <fstream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <stdio.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "graph.hpp"
#include "input.hpp"

///
/// Namespaces
///
namespace pt = boost::posix_time;

///
/// Generate a random graph
///
void generate_random_graph(GraphData *graph, int num_vertices, int neighbors_per_vertex) {
    graph->vertex_count = num_vertices;
    graph->vertex_array = static_cast<int *>(malloc(graph->vertex_count * sizeof(int)));
    graph->edge_count = num_vertices * neighbors_per_vertex;
    graph->edge_array = static_cast<int *>(malloc(graph->edge_count * sizeof(int)));
    graph->weight_array = static_cast<float *>(malloc(graph->edge_count * sizeof(float)));

    for (int i = 0; i < graph->vertex_count; ++i) {
        graph->vertex_array[i] = i * neighbors_per_vertex;
    }

    for (int i = 0; i < graph->edge_count; ++i) {
        graph->edge_array[i] = (rand() % graph->vertex_count);
        graph->weight_array[i] = static_cast<float>(rand() % 1000) / 1000.0f;
    }
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

///
/// Gets the id of the first device from the context (from the NVIDIA SDK)
///
cl_device_id get_first_dev(cl_context cx_GPU_context) {
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

void check_error_file_line(int err_num, int expected, const char *file, const int line_number) {
    if (err_num != expected) {
        std::cerr << "Error at line " << line_number << " in file " << file << std::endl;
        exit(1);
    }
}

#define check_error(a, b) check_error_file_line(a, b, __FILE__, __LINE__)

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

///
/// Load and build an OpenCL program from source file
/// \param gpu_context GPU context on which to load and build the program
/// \param file_name   File name of source file that holds the kernels
/// \return            Handle to the program
///
cl_program load_and_build_program(cl_context gpu_context, const char *file_name) {
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

///
/// Round the local work size up to the next multiple of the size
///
int round_work_size_up(int group_size, int global_size) {
    int remainder = global_size % group_size;
    if (remainder == 0) return global_size;
    else return global_size + group_size - remainder;
}

///
/// Allocate memory for input CUDA buffers and copy the data into device memory
///
void allocate_OCL_buffers(cl_context gpu_context, cl_command_queue command_queue,
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

///
/// Initialize OpenCL buffers for single run of Dijkstra
///
void initialize_OCL_buffers(cl_command_queue command_queue, cl_kernel initialize_kernel,
                            GraphData *graph, size_t max_work_group_size) {
    cl_int err_num;
    // Set # of work items in work group and total in 1 dimensional range
    size_t local_work_size = max_work_group_size;
    size_t global_work_size = round_work_size_up(local_work_size, graph->vertex_count);

    err_num = clEnqueueNDRangeKernel(command_queue, initialize_kernel, 1, NULL,
                                     &global_work_size, &local_work_size, 0, NULL, NULL);
    check_error(err_num, CL_SUCCESS);
}

///
/// Check whether the mask array is empty. This tells the algorithm whether
/// it needs to continue running or not
///
bool mask_array_empty(int *mask_array, int count) {
    for (int i = 0; i < count; ++i) {
        if (mask_array[i] == 1) return false;
    }
    return true;
}

#define NUM_ASYNCHRONOUS_ITERATIONS 10  // Number of async loop iterations before attempting to read results back

///
/// Run Dijkstra's shortest path on the GraphData provided to this function.
///
void run_Dijkstra(cl_context context, cl_device_id device_id, GraphData *graph,
                  int *source_vertices, float *out_result_costs, int num_results) {
    // Create command queue
    cl_int err_num;
    cl_command_queue command_queue;
    command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err_num);
    check_error(err_num, CL_SUCCESS);

    // Program handle
    cl_program program = load_and_build_program(context, "include/kernel.cl");
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
    allocate_OCL_buffers(context, command_queue, graph, &vertex_array_device,
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

    // 3 set below in loop
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

    int *maskArrayHost = (int*) malloc(sizeof(int) * graph->vertex_count);

    for (int i = 0; i < num_results; i++ ) {
        err_num |= clSetKernelArg(initialize_buffers_kernel, 3, sizeof(int), &source_vertices[i]);
        check_error(err_num, CL_SUCCESS);

        // Initialize mask array to false, C and U to infiniti
        initialize_OCL_buffers(command_queue, initialize_buffers_kernel, graph, max_work_group_size);

        // Read mask array from device -> host
        cl_event read_done;
        err_num = clEnqueueReadBuffer(command_queue, mask_array_device, CL_FALSE, 0, sizeof(int) * graph->vertex_count,
                                      maskArrayHost, 0, NULL, &read_done);
        check_error(err_num, CL_SUCCESS);
        clWaitForEvents(1, &read_done);

        while(!mask_array_empty(maskArrayHost, graph->vertex_count))
        {

            // In order to improve performance, we run some number of iterations
            // without reading the results.  This might result in running more iterations
            // than necessary at times, but it will in most cases be faster because
            // we are doing less stalling of the GPU waiting for results.
            for(int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; asyncIter++)
            {
                size_t localWorkSize = max_work_group_size;
                size_t global_work_size = round_work_size_up(localWorkSize, graph->vertex_count);

                // execute the kernel
                err_num = clEnqueueNDRangeKernel(command_queue, ssspKernel1, 1, 0, &global_work_size, &localWorkSize,
                                               0, NULL, NULL);
                check_error(err_num, CL_SUCCESS);

                err_num = clEnqueueNDRangeKernel(command_queue, ssspKernel2, 1, 0, &global_work_size, &localWorkSize,
                                               0, NULL, NULL);
                check_error(err_num, CL_SUCCESS);
            }
            err_num = clEnqueueReadBuffer(command_queue, mask_array_device, CL_FALSE, 0, sizeof(int) * graph->vertex_count,
                                         maskArrayHost, 0, NULL, &read_done);
            check_error(err_num, CL_SUCCESS);
            clWaitForEvents(1, &read_done);
        }

        // Copy the result back
        err_num = clEnqueueReadBuffer(command_queue, cost_array_device, CL_FALSE, 0, sizeof(float) * graph->vertex_count,
                                     &out_result_costs[i * graph->vertex_count], 0, NULL, &read_done);
        check_error(err_num, CL_SUCCESS);
        clWaitForEvents(1, &read_done);
    }

    free(maskArrayHost);

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

///
/// main
///
int main(int argc, char **argv) {
    bool do_GPU = false;
    bool do_org = false;
    bool do_ref = false;
    int num_sources = 100;
    int generate_verts = 100000;
    int generate_edges_per_vert = 10;

    if(!parse_command_line_args(argc, argv, do_GPU, do_org, do_ref,
                            &num_sources, &generate_verts, &generate_edges_per_vert))
        exit(1);

    cl_platform_id platform;
    cl_context gpu_context;
    cl_context cpu_context;
    cl_int err_num;

    // First, select an OpenCL platform to run on. For this example, we
    // simply choose the first available platform. Normally, you would
    // query for all available platform an select the most appropriate one.
    cl_uint num_platforms;
    err_num = clGetPlatformIDs(1, &platform, &num_platforms);
    std::cout << "Number of OpenCL Platforms: " << num_platforms << std::endl;
    if (err_num != CL_SUCCESS || num_platforms <= 0) {
        std::cout << "Failed to find any OpenCL platforms." << std::endl;
        return 1;
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

    // Allocate memory for arrays
    GraphData graph;
    generate_random_graph(&graph, generate_verts, generate_edges_per_vert);

    std::cout << "Vertex Count: " << graph.vertex_count << std::endl;
    std::cout << "Edge Count: " << graph.edge_count << std::endl;

    std::vector<int> source_vertices;

    int *source_vert_array = static_cast<int *>(malloc(sizeof(int) * num_sources));
    for (int source = 0; source < num_sources; ++source) {
        source_vert_array[source] = (source % graph.vertex_count);
    }

    float *results = static_cast<float *>(malloc(sizeof(float) * num_sources * graph.vertex_count));

    // Run Dijkstra's algorithm
    pt::ptime start_time_CPU = pt::microsec_clock::local_time();
    if (do_GPU)
        run_Dijkstra(gpu_context, get_max_flops_dev(gpu_context), &graph, source_vert_array,
                     results, num_sources);

    pt::time_duration time_cput = pt::microsec_clock::local_time() - start_time_CPU;

    std::cout << "Running time: " << time_cput.abs() << std::endl;

    free(source_vert_array);
    free(results);

    clReleaseContext(gpu_context);
}
