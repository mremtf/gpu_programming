#include <vector>

#include "../include/transpose.hpp"

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
