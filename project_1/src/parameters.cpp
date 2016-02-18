#include "parameters.hpp"
#include <boost/program_options.hpp>
#include <boost/version.hpp>
#include <exception>
#include <iostream>

namespace po = boost::program_options;

/*
using options_t = struct {
    bool validate;
    bool multi;
    unsigned blocks, threads;
    double utilization;
};
*/

options_t process_params(int argc, char *argv[]) {
    po::options_description desc("Options");
    // clang-format off
    // My beautiful program options get wrecked by formatting
    #if ((BOOST_VERSION / 100) % 1000) < 42
    desc.add_options()
        ("blocks,b",po::value<unsigned>()->default_value(32),"Block count")
        ("threads,t",po::value<unsigned>()->default_value(32),"Threads per block")
        ("mem_utilization,u",po::value<double>()->default_value(.5),"Device memory utilization")
        ("multigpu,m","Multi-GPU")
        ("validate,v","Validate GPU results");
    #else
    desc.add_options()
        ("blocks,b",po::value<unsigned>()->required(),"Block count")
        ("threads,t",po::value<unsigned>()->required(),"Threads per block")
        ("mem_utilization,u",po::value<double>()->required(),"Device memory utilization")
        ("multigpu,m","Multi-GPU")
        ("validate,v","Validate GPU results");
    #endif
    // clang-format on
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    try {
        po::notify(vm);
    } catch (std::exception &e) {
        std::cout << "Bad parameters!\n" << desc << std::endl;
        exit(1);
    }

    return options_t{vm.count("validate") == 1, vm.count("multigpu") == 1, vm["blocks"].as<unsigned>(),
                     vm["threads"].as<unsigned>(), vm["mem_utilization"].as<double>()};
}
