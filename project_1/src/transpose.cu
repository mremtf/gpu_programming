#include <vector>
#include <iostream>
#include <stdexcept>

#include "transpose.hpp"
#include "parameters.hpp"
#include "device_queries.hpp"
#include "shared_utilities.hpp"
#include "timer.hpp"

bool cpu_transpose(const std::vector<float>& in, std::vector<float>& out, const size_t N, const size_t M) {
	if (in.empty() || N == 0 || M == 0) {
		return false;
	}

	for(unsigned n = 0; n<N*M; n++) {
		const size_t row = n/N;
		const size_t col = n%N;
		out[n] = in[M*col + row];
	}
	return true;
}

void __global__ transpose_global (float *in, float *out, const unsigned W, unsigned step, 
		const unsigned total, const unsigned fix_position, unsigned fix_step) {

	  unsigned position = blockDim.x * blockIdx.x + threadIdx.x;
		//printf("pos = %u\n", position);
    position *= step;
    // printf("%d\t%d\t%d\t%d\n", blockDim.x, blockIdx.x, threadIdx.x, position);
    // This is a really dumb edge case clearly used to break the code, but
    // hell if I'm missing points for not catching when you request more threads than elements!
    if (position < total) {
        // Interesting thing to test
        // ternary here vs if. Only divergence should be last warp in last block
        // But the ternary will probably slow down everything?
        // It would avoid a warp divergence, though!
        if (position == fix_position) {
            step = fix_step;
        }
        in += position;
				unsigned y = 0; //floor( (float) position/ (float)W);
				unsigned x = 0; //position - (y * W);  
        for (int i = 0; i < step; ++i, ++position, ++in) {
            // printf("%p %p %i %i %f %f\n", a, b, position, i, *a, *b);
						y = position / W;
						x = position - (y * W); 
            //printf ("%u %u %u %u\n", position, x,y,x*W +y);
						out[x * W + y] = *in;
        }
    }
}

void __global__ transpose_shared (float *in, float *out, const unsigned W, unsigned step, 
		const unsigned total, const unsigned fix_position, unsigned fix_step) {

		extern __shared__ float tile[];

	  unsigned position = blockDim.x * blockIdx.x + threadIdx.x;
		//printf("pos = %u\n", position);
    //position *= step;
	

    // printf("%d\t%d\t%d\t%d\n", blockDim.x, blockIdx.x, threadIdx.x, position);
    if (position < total) {
						// load shared memory
      in += position;
			unsigned y = 0; //floor( (float) position/ (float)W);
			unsigned x = 0; //position - (y * W);  

			for (unsigned s = 0; s < step; ++s, position+=blockDim.x, in+=blockDim.x) {	
				
				tile[threadIdx.x] = *in;
				__syncthreads();

        // printf("%p %p %i %i %f %f\n", a, b, position, i, *a, *b);
				y = position/ W;
				x = position - (y * W); 
        printf ("%u %u %u %u %u\n", position, x,y,x*W +y, threadIdx.x);
				out[x * W + y] = tile[threadIdx.x];
       
			}
			printf("%i %i\n", position, threadIdx.x);
      if (position == fix_position) {
        		for (unsigned i = 0; i < fix_step; ++i, ++position, ++in) {
            	// printf("%p %p %i %i %f %f\n", a, b, position, i, *a, *b);
							y = position/ W;
							x = position - (y * W); 
            	printf ("LEFT OVER: %u %u %u %u\n", position, x,y,x*W +y);
							out[x * W + y] = *in;
        	}
      }

		}
} 

using device_config_t = struct {
    int device;
    void *matrix_in_device, *matrix_out_device;  // vecs gets summed into a
		unsigned matrix_width;
    unsigned step;
    unsigned fix_position;  // UINT_MAX
    unsigned fix_step;
};


void launch_kernels_and_report(const options_t &opts) {
    const int threads         = opts.threads;
    const int blocks          = opts.blocks;
    const bool validate       = opts.validate;
    const double util         = opts.utilization;
    const size_t thread_total = blocks * threads;

    if (threads == 0 || blocks == 0) {
        throw std::runtime_error("Thread/Block count of 0!");
    }

    std::vector<int> devices = get_devices();
		
		if (!devices.size()) {
			std::cout << "No devices" << std::endl;
			return;
		}
	
    size_t n_elems = get_global_mem(0) / sizeof(float) * util / 2.0; // get the total number of elements
		size_t matrix_n = floor(sqrt((float)n_elems)); // get the width of the matrix
		n_elems = matrix_n * matrix_n; // sqaure the values to make it matrix

		std::cout << "N = " << matrix_n << std::endl;

    device_config_t config;	
    config.device = 0;
		config.matrix_width = matrix_n;
    auto dim_pair = get_dims(config.device);
    if (dim_pair.first < threads || dim_pair.second < blocks) {
    	throw std::runtime_error("Block/thread count outside device dims!");
    }
    config.step = n_elems / thread_total;
		const bool offset_needed = (config.step * thread_total) != n_elems;
    if (config.step == 0) {
    	std::cout << "More threads than values! Rude!" << std::endl;
      // with a very low mem utilization (read: testing)
      // it will end up with a step of 0 if you get total_threads over n_elem
      // So I guess hardcode 1 and nop anything off the end of the vector
      config.step         = 1;
      config.fix_position = UINT_MAX;
      config.fix_step     = 1;

    } else {
    	
      if (offset_needed) {
      	config.fix_position = config.step * (thread_total - 1);
        config.fix_step     = config.step + (n_elems - (config.step * thread_total));
      } 
			else {
                config.fix_position = UINT_MAX;        // should never trigger
                config.fix_step     = config.step;  // but just in case
      }
    }

		timer gpu_total, gpu_execute;

		std::cout << "Dev: " << config.device << " Step: " << config.step << " Fix_P: " << config.fix_position
              << " Fix_s: " << config.fix_step << " Threads: " << thread_total
              << " Val total: " << n_elems << std::endl;

    std::vector<float> in = generate_vector(n_elems);
   	std::vector<float> out = generate_vector(n_elems);
    std::vector<float> c = std::vector<float>(n_elems);
		std::vector<float> c_shared = std::vector<float>(n_elems);
		

    if (cudaSetDevice(config.device) != cudaSuccess) {
    	throw std::runtime_error("could not select device!");
    }

    gpu_total.begin();

    if (cudaMalloc(&config.matrix_in_device, n_elems * sizeof(float)) != cudaSuccess
    			|| cudaMalloc(&config.matrix_out_device, n_elems * sizeof(float)) != cudaSuccess) {
    	throw std::runtime_error("Failed to malloc vector!");
    }

    if (cudaMemcpy(config.matrix_in_device, in.data(), n_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
    	throw std::runtime_error("Failed to copy data to device!");
    }


		/*
		* GLOBAL MEMORY TIMING
		*/
    gpu_execute.begin();

    transpose_global<<<blocks, threads>>>((float *) config.matrix_in_device, (float *) config.matrix_out_device, 
																						 config.matrix_width, config.step, n_elems, config.fix_position,
                                             config.fix_step);

    if (cudaDeviceSynchronize() != cudaSuccess) {
    	throw std::runtime_error("Sync issue! (Launch failure?)");
    }

    gpu_execute.end();

    std::cout << "GLOBAL MEMORY GPU_" << config.device << " time: " << gpu_execute.ms_elapsed() << std::endl;	

    if (cudaMemcpy(c.data(), config.matrix_out_device, n_elems * sizeof(float), cudaMemcpyDeviceToHost)
            != cudaSuccess) {
   		throw std::runtime_error("Could not copy data back!");
    }

		if (cudaMemset(config.matrix_out_device, 0, n_elems * sizeof(float))
            != cudaSuccess) {
   		throw std::runtime_error("Could not memset out matrix!");
    }

		/*
		* SHARED MEMORY TIMING
		*/
		// NEED IN-CASE FOR INTERLEAVING -- IMPORTANTE
		/*if (offset_needed) {
      config.fix_position = config.step * (thread_total);
      config.fix_step     = (n_elems - (config.step * thread_total));
		std::cout << "Dev: " << config.device << " Step: " << config.step << " Fix_P: " << config.fix_position
              << " Fix_s: " << config.fix_step << " Threads: " << thread_total
              << " Val total: " << n_elems << std::endl;
    } 

		gpu_execute.begin();

    transpose_shared<<<blocks, threads, threads*sizeof(float)>>>((float *) config.matrix_in_device, (float *) config.matrix_out_device, 
																						 config.matrix_width, config.step, n_elems, config.fix_position,
                                             config.fix_step);
		
    if (cudaDeviceSynchronize() != cudaSuccess) {
			printf("CUDA ERROR = %s\n", cudaGetErrorString(cudaGetLastError()));
    	throw std::runtime_error("Sync issue! (Launch failure?)");
    }

    gpu_execute.end();


    if (cudaMemcpy(c_shared.data(), config.matrix_out_device, n_elems * sizeof(float), cudaMemcpyDeviceToHost)
            != cudaSuccess) {
   		throw std::runtime_error("Could not copy data back!");
    }*/


    cudaFree(config.matrix_in_device);
    cudaFree(config.matrix_out_device);
		gpu_total.end();

    //std::cout << "SHARED MEMORY GPU_" << config.device << " time: " << gpu_execute.ms_elapsed() << std::endl;	

		if (validate) {
    	timer cpu_time;
      cpu_time.begin();
			std::vector<float> cpu_res = std::vector<float>(n_elems);
      cpu_transpose(in,cpu_res,config.matrix_width,config.matrix_width);
      cpu_time.end();
      std::cout << "CPU time: " << cpu_time.ms_elapsed() << " ms" << std::endl;
      if (!check_equal(c, cpu_res)) {
				std::cout << "FAILED LOSER GLOBAL MEMORY" << std::endl;
				std::cout << "INPUT " << std::endl;
				for (unsigned r = 0; r < matrix_n;++r) {
					for (unsigned x = 0; x < matrix_n; ++x) {
						std::cout << in[r * matrix_n + x] << " ";
					}
					std::cout << std::endl;
				}

				std::cout << std::endl <<"CPU RESULT" << std::endl;
				for (unsigned r = 0; r < matrix_n;++r) {
					for (unsigned x = 0; x < matrix_n; ++x) {
						std::cout << cpu_res[r * matrix_n + x] << " ";
					}
					std::cout << std::endl;
				}
				std::cout << std::endl <<"GPU RESULT" << std::endl;
				for (unsigned r = 0; r < matrix_n;++r) {
					for (unsigned x = 0; x < matrix_n; ++x) {
						std::cout << c[r * matrix_n + x] << " ";
					}
					std::cout << std::endl;
				}
    	}
      /*if (!check_equal(c_shared, cpu_res)) {
				std::cout << "FAILED LOSER: SHARED MEMORY" << std::endl;
				std::cout << "INPUT " << std::endl;
				for (unsigned r = 0; r < matrix_n;++r) {
					for (unsigned x = 0; x < matrix_n; ++x) {
						std::cout << in[r * matrix_n + x] << " ";
					}
					std::cout << std::endl;
				}

				std::cout << std::endl <<"CPU RESULT" << std::endl;
				for (unsigned r = 0; r < matrix_n;++r) {
					for (unsigned x = 0; x < matrix_n; ++x) {
						std::cout << cpu_res[r * matrix_n + x] << " ";
					}
					std::cout << std::endl;
				}
				std::cout << std::endl <<"GPU RESULT" << std::endl;
				for (unsigned r = 0; r < matrix_n;++r) {
					for (unsigned x = 0; x < matrix_n; ++x) {
						std::cout << c_shared[r * matrix_n + x] << " ";
					}
					std::cout << std::endl;
				}
			}*/
		}
    return;
}
