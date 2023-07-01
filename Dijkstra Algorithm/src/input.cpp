#include <iostream>
#include <boost/program_options.hpp>

#include "input.hpp"

namespace po = boost::program_options;

bool parse_command_line_args(int argc, char **argv,
                             bool &do_GPU, bool &do_org,
                             bool &do_ref,
                             int *source_verts,
                             int *generate_verts, int *generate_edges_per_vert) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help",                        "Produce help message")
        ("gpu",                         "Run single GPU version of algorithm")
        ("org",                         "Run original version of algorithm")
        ("ref",                         "Run reference version of algorithm")
        ("sources",   po::value<int>(), "Number of source vertices to search from (default: 100)")
        ("verts",     po::value<int>(), "Number of vertices in randomly generated graph (default: 100000)")
        ("edges",     po::value<int>(), "Number of edges per vertex in randomly generated graph (default: 10)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") || argc == 1) {
        std::cout << desc << "\n";
        return false;
    }

    // Parse options
    if (vm.count("gpu"))        do_GPU = true;
    if (vm.count("org"))        do_org = true;
    if (vm.count("ref"))        do_ref = true;
    if (vm.count("sources"))    *source_verts = vm["sources"].as<int>();
    if (vm.count("verts"))      *generate_verts = vm["verts"].as<int>();
    if (vm.count("edges"))      *generate_edges_per_vert = vm["edges"].as<int>();

    return true;
}
