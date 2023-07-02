#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <stdio.h>

#include "graph.hpp"
#include "input.hpp"
#include "opencl_sol.hpp"

namespace pt = boost::posix_time;

int main(int argc, char **argv) {
    Input inp;
    inp.parse_command_line_args(argc, argv);

    OpenCLSolution ocl_sol;

    // Allocate memory for arrays
    GraphData graph;
    generate_random_graph(&graph, inp.get_generate_verts(), inp.get_generate_edges_per_vert());

    std::cout << "Vertex Count: " << graph.vertex_count << std::endl;
    std::cout << "Edge Count: " << graph.edge_count << std::endl;

    std::vector<int> source_vertices;

    int *source_vert_array = static_cast<int *>(malloc(sizeof(int) * inp.get_num_sources()));
    for (int source = 0; source < inp.get_num_sources(); ++source) {
        source_vert_array[source] = (source % graph.vertex_count);
    }

    float *results = static_cast<float *>(malloc(sizeof(float) * inp.get_num_sources() * graph.vertex_count));

    // Run Dijkstra's algorithm
    pt::ptime start_time_CPU = pt::microsec_clock::local_time();
    if (inp.get_do_GPU())
        ocl_sol.run_Dijkstra(&graph, source_vert_array,
                             results, inp.get_num_sources());

    pt::time_duration time_cput = pt::microsec_clock::local_time() - start_time_CPU;

    std::cout << "Running time: " << time_cput.abs() << std::endl;

    free(source_vert_array);
    free(results);
}
