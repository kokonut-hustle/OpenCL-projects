#pragma once

// Data Structure for Dijkstra's Algorithm

typedef struct
{
    // (V) This contains a pointer to the edge list for each vertex
    int *vertex_array;

    // Vertex count
    int vertex_count;

    // (E) This contains pointers to the vertices that each edge
    // is attached to
    int *edge_array;

    // Edge count
    int edge_count;

    // (W) Weight array
    float *weight_array;
} GraphData;

void generate_random_graph(GraphData *graph, int num_vertices, int neighbors_per_vertex);
