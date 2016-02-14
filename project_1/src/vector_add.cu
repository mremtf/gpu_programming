#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>

#include "vector_add.hpp"
#include "parameters.hpp"
#include "device_queries.hpp"
#include "shared_utilities.hpp"

using std::vector;

vector<float> cpu_addition(const vector<float> &a, const vector<float> &b) {
    vector<float> results(a);
    std::transform(results.begin(), results.end(), b.cbegin(), results.begin(), std::plus<float>());
    return a;
}

/*
calculate step, calculate final index, if step doesn't work, specify final index and special step

if special step doesn't exist, FFFFFFFF it. only warp divergence will be in one warp in final block.

ideally, last block on last device
*/

void launch_kernels_and_report(const options_t &opts) {
    return;
}
