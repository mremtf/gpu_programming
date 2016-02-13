#include "parameters.hpp"
#include "device_queries.hpp"
#include "timer.hpp"
#include "vector_add.hpp"

#include <iostream>

int main(int argc, char **argv) {
    simple_timer a;
    a.begin();
    options_t prog_opts = process_params(argc, argv);
    a.end();
    std::cout << "Took " << a.ms_elapsed() << std::endl;

    auto devices = get_devices();
    std::cout << "Got " << devices.size() << " devices.\n";
    for (const auto i : devices) {
        std::cout << "Device " << i << " Global mem: " << get_global_mem(i) << std::endl;
    }

    std::cout << "Get vec of 5 rand: ";
    for (const auto num : generate_vector(5)) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}