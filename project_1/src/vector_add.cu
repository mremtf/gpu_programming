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

using device_config_t = struct {
    int device;
    void *vec_a_device, *vec_b_device;  // vecs gets summed into a
    size_t step;
    size_t final_position;  // SIZE_MAX if it doesn't exist
    size_t final_step;
};

void launch_kernels_and_report(const options_t &opts) {
    const unsigned threads = opts.threads;
    const unsigned blocks  = opts.blocks;
    const bool validate    = opts.validate;
    const bool multi       = opts.multi;
    const double util      = opts.utilization;

    std::vector<int> devices = get_devices();
    
}
