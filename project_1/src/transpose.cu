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

__global__ transpose_shared(float* in_matrix, float* out_matrix, size_t rows, size_t cols) {

}


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

		float mem = get_global_mem(devices[i]) / sizeof(float) * util / 2.0;

		timer gpu_total, gpu_execute;

		std::cout << "Dev: " << config[i].device << " Step: " << config[i].step << " Fix_P: " << config[i].fix_position
              << " Fix_s: " << config[i].fix_step << " Threads: " << thread_total
              << " Val total: " << float_vec_size[i] << std::endl;

    std::vector<float> in = generate_vector(float_vec_size[i]);
   	std::vector<float> out = generate_vector(float_vec_size[i]);
    std::vector<float> c = std::vector<float>(float_vec_size[i]);

    if (cudaSetDevice(config[i].device) != cudaSuccess) {
    	throw std::runtime_error("could not select device!");
    }

    gpu_total.begin();

    if (cudaMalloc(&config[i].vec_a_device, float_vec_size[i] * sizeof(float)) != cudaSuccess
    			|| cudaMalloc(&config[i].vec_b_device, float_vec_size[i] * sizeof(float)) != cudaSuccess) {
        throw std::runtime_error("Failed to malloc vector!");
    }

    if (cudaMemcpy(config[i].vec_a_device, a.data(), float_vec_size[i] * sizeof(float), cudaMemcpyHostToDevice)
    		!= cudaSuccess
        || cudaMemcpy(config[i].vec_b_device, b.data(), float_vec_size[i] * sizeof(float), cudaMemcpyHostToDevice)
                   != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to device!");
    }

    gpu_execute.begin();

    cuda_vector_add<<<blocks, threads>>>((float *) config[i].vec_a_device, (float *) config[i].vec_b_device,
                                             config[i].step, float_vec_size[i], config[i].fix_position,
                                             config[i].fix_step);

    if (cudaDeviceSynchronize() != cudaSuccess) {
    	throw std::runtime_error("Sync issue! (Launch failure?)");
    }

    gpu_execute.end();

    if (cudaMemcpy(c.data(), config[i].vec_a_device, float_vec_size[i] * sizeof(float), cudaMemcpyDeviceToHost)
            != cudaSuccess) {
   		throw std::runtime_error("Could not copy data back!");
    }

    cudaFree(config[i].vec_a_device);
    cudaFree(config[i].vec_b_device);
		gpu_total.end();

    std::cout << "GPU_" << config[i].device << " time: " << gpu_total.ms_elapsed()	

    return;
}
