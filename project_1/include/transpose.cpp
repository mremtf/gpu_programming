#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_

#include <vector>

bool cpu_transpose(const std::vector<float>& in, const std::vector<float>& out, const size_t N, const size_t M);

#endif
