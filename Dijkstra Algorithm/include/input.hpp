#pragma once

bool parse_command_line_args(int argc, char **argv,
                             bool &do_GPU, bool &do_org,
                             bool &do_ref,
                             int *source_verts,
                             int *generate_verts, int *generate_edges_per_vert);
