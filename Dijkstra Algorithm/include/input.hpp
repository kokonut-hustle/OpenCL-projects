#pragma once

class Input {
public:
    Input() : do_GPU(false), do_org(false), do_ref(false),
              num_sources(100), generate_verts(100000),
              generate_edges_per_vert(10) {
    }

    void parse_command_line_args(int argc, char **argv);

    bool get_do_GPU() const {
        return do_GPU;
    }

    int get_num_sources() const {
        return num_sources;
    }

    int get_generate_verts() const {
        return generate_verts;
    }

    int get_generate_edges_per_vert() const {
        return generate_edges_per_vert;
    }

private:
    bool do_GPU;
    bool do_org;
    bool do_ref;
    int num_sources;
    int generate_verts;
    int generate_edges_per_vert;
};
