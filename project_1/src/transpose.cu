#include <vector>

#include "transpose.hpp"
#include "parameters.hpp"
#include "device_queries.hpp"
#include "shared_utilities.hpp"

bool cpu_transpose(const std::vector<float>& in, const std::vector<float>& out, const size_t N, const size_t M) {
	const float *src = src.data[0];
	float dst = out.data[0];
	if (!src || !dst || N == 0 || M == 0) {
		return false;
	}

	for(auto n = 0; n<N*M; n++) {
		const size_t row = n/N;
		const size_t col = n%N;
		dst[n] = src[M*col + row];
	}
	return true;
}

__global__ transpose_global (float *in, float *out, const unsigned W, const unsigned step, 
		const unsigned total, const unsigned fix_position, unsigned fix_step) {

	  unsigned position = blockDim.x * blockIdx.x + threadIdx.x;
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
        for (int i = 0; i < step; ++i, ++position) {
            // printf("%p %p %i %i %f %f\n", a, b, position, i, *a, *b);
						unsigned y = floor( (float) position/ (float)W);
						unsigned x = position - (y * W); 
            out[x * W + y] += *in;
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
    const bool multi          = opts.multi;
    const double util         = opts.utilization;
    const size_t thread_total = blocks * threads;

    if (threads == 0 || blocks == 0) {
        throw std::runtime_error("Thread/Block count of 0!");
    }

    std::vector<int> devices = get_devices();
		
		if (devices.size()) {
			std::cout << "No devices" << std::endl;
			return;
		}
	
    size_t mem_size = get_global_mem(0) / sizeof(float) * util / 2.0;
		size_t matrix_n = floor(sqrt((float)mem_size));
    // number of total floats, get the utilization, div in two because a + b
    // resulting size is the size for vectors a and b

    // Instead of making a giant contiguous vector and serving out slices to the devices
    // I'm just going to make smaller ones since there's no real difference

    device_config_t config;
    auto dim_pair = get_dims(config);
    if (dim_pair.first < threads || dim_pair.second < blocks) {
    	throw std::runtime_error("Block/thread count outside device dims!");
    }
    config.device = 0;
    config.step = mem_size / thread_total;
    if (config.step == 0) {
    	std::cout << "More threads than values! Rude!" << std::endl;
      // with a very low mem utilization (read: testing)
      // it will end up with a step of 0 if you get total_threads over n_elem
      // So I guess hardcode 1 and nop anything off the end of the vector
      config.step         = 1;
      config.fix_position = UINT_MAX;
      config.fix_step     = 1;
    } else {
    	const bool offset_needed = (config.step * thread_total) != mem_size;
      if (offset_needed) {
      	config.fix_position = config.step * (thread_total - 1);
        config.fix_step     = config.step + (mem_size - (config.step * thread_total));
      } 
			else {
                config.fix_position = UINT_MAX;        // should never trigger
                config.fix_step     = config.step;  // but just in case
      }
    }

		timer gpu_total, gpu_execute;

		std::cout << "Dev: " << config.device << " Step: " << config.step << " Fix_P: " << config.fix_position
              << " Fix_s: " << config.fix_step << " Threads: " << thread_total
              << " Val total: " << mem_size << std::endl;

    std::vector<float> in = generate_vector(mem_size);
   	std::vector<float> out = generate_vector(mem_size);
    std::vector<float> c = std::vector<float>(mem_size);

    if (cudaSetDevice(config.device) != cudaSuccess) {
    	throw std::runtime_error("could not select device!");
    }

    gpu_total.begin();

    if (cudaMalloc(&config.matrix_in_device, mem_size * sizeof(float)) != cudaSuccess
    			|| cudaMalloc(&config.matrix_out_device, mem_size * sizeof(float)) != cudaSuccess) {
    	throw std::runtime_error("Failed to malloc vector!");
    }

    if (cudaMemcpy(config.matrix_in_device, a.data(), mem_size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
    	throw std::runtime_error("Failed to copy data to device!");
    }

    gpu_execute.begin();

    cuda_vector_add<<<blocks, threads>>>((float *) config.matrix_in_device, (float *) config.matrix_out_device, 
																						 config.matrix_width, config.step, mem_size, config.fix_position,
                                             config.fix_step);

    if (cudaDeviceSynchronize() != cudaSuccess) {
    	throw std::runtime_error("Sync issue! (Launch failure?)");
    }

    gpu_execute.end();

    if (cudaMemcpy(c.data(), config.matrix_in_device, mem_size * sizeof(float), cudaMemcpyDeviceToHost)
            != cudaSuccess) {
   		throw std::runtime_error("Could not copy data back!");
    }

    cudaFree(config.matrix_in_device);
    cudaFree(config.matrix_out_device);
		gpu_total.end();

    std::cout << "GPU_" << config.device << " time: " << gpu_total.ms_elapsed()	

    return;
}
