#include "parameters.hpp"
#include "vector_add.hpp"

#include <iostream>

int main(int argc, char **argv) {
    
    options_t prog_opts = process_params(argc, argv);

    launch_kernels_and_report(prog_opts);

}