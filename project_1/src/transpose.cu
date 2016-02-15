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

__global__ transpose_global (const float* in_matrix, float* out_matrix, const size_t rows, const size_t cols) {
	const size_t x = blockIdx.x * TILE_DIM + threadIdx.x;
  const size_t y = blockIdx.y * TILE_DIM + threadIdx.y;
	const width = gridDim.x * TILE_DIM;

		  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
				    odata[(y+j)*width + x] = idata[(y+j)*width + x];
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


    return;
}
