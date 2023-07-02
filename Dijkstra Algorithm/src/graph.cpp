#include <stdlib.h>

#include "graph.hpp"

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
