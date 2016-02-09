#include "../include/parameters.hpp"
#include "../include/timer.hpp"

int main(int argc, char **argv) {
    simple_timer a;
    a.begin();
    options_t prog_opts = process_params(argc, argv);
    a.end();
    std::cout << "Took " << a.ms_elapsed() << std::endl;
}