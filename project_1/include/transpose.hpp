#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_

#include "parameters.hpp"
#include <vector>
#include <cstddef>

bool cpu_transpose(const std::vector<float>& in, const std::vector<float>& out, const size_t N, const size_t M);

void launch_kernels_and_report(const options_t &opts);

#endif
