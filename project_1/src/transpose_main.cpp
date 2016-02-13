#include "../include/parameters.hpp"
#include "../include/device_queries.hpp"
#include "../include/timer.hpp"

#include "../include/transpose.hpp"

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

		std::vector<float> input_matrix = generate_vector();
		if (input_matrix.empty()) {
			std::cout << "No memory for allocation of input matrix" << std::end;
			return -1;
		}

		std::vector<float> output_matrix = generate_vector();
		if (output_matrix.empty())	
			std::cout << "No memory for allocation of input matrix" << std::end;
			return -1;
		}

		if(cpu_transpose(input_matrix,output_matrix)) {
			std::cout << "Failed to cpu transpose" << std::endl;
			return -1;
		}

		

}
