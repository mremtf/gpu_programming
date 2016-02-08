#ifndef _PARAMETERS_HPP__
#define _PARAMETERS_HPP__

#include <boost/program_options>
#include <iostream>

namespace po = boost::program_options;

po::variables_map process_variables(int argc, char *argv[]) {
    po::options_description desc("Options");
    // clang-format off
    // My beautiful program options get wrecked by formatting
	desc.add_options()
		("blocks,blk",po::value<unsigned>()->required(),"Block count")
		("threads,tpb",po::value<unsigned>()->required(),"Threads per block")
		("mem_utilization,mu",po::value<double>()->required(),"Device memory utilization")
		("multigpu,mg","Multi-GPU")
		("validate,v","Validate GPU results");
    // clang-format on
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    try {
        po::notify(vm);
    } catch (std::exception &e) {
        std::cout <<"Bad parameters!\n"<<desc<<std::endl;
        exit(1);
    }
    return vm;
}

#endif