#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdint>
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
    std::vector<float> a, b;
    size_t step;
    size_t fix_position;  // SIZE_MAX if it doesn't exist
    size_t fix_step;
};

void launch_kernels_and_report(const options_t &opts) {
    const unsigned threads    = opts.threads;
    const unsigned blocks     = opts.blocks;
    const bool validate       = opts.validate;
    const bool multi          = opts.multi;
    const double util         = opts.utilization;
    const size_t thread_total = blocks * threads;

    if (threads == 0 || blocks == 0) {
        throw std::runtime_error("Thread/Block count of 0!");
    }

    std::vector<int> devices = get_devices();
    if (!multi) {
        devices.resize(1);
    }
    const size_t num_devices = devices.size();

    std::vector<size_t> float_vec_size(num_devices);
    for (int i = 0; i < num_devices; ++i) {
        float_vec_size[i] = get_global_mem(devices[i]) / sizeof(float) * util / 2.0;
        // number of total floats, get the utilization, div in two because a + b
        // resulting size is the size for vectors a and b
    }

    // Instead of making a giant contiguous vector and serving out slices to the devices
    // I'm just going to make smaller ones since there's no real difference

    std::vector<device_config_t> config(num_devices);
    for (int i = 0; i < num_devices; ++i) {
        auto dim_pair = get_dims(devices[i]);
        if (dim_pair.first < threads || dim_pair.second < blocks) {
            throw std::runtime_error("Block/thread count outside device dims!");
        }
        config[i].device   = devices[i];
        config[i].a        = generate_vector(float_vec_size[i]);
        config[i].b        = generate_vector(float_vec_size[i]);
        config[i].step     = float_vec_size[i] / thread_total;
        const bool offset_needed = (config[i].step * thread_total) != float_vec_size[i];
        if (offset_needed) {
        	config[i].fix_position = step * (thread_total - 1);
        	config[i].fix_step = config[i].step + (float_vec_size[i] - (config[i].step * thread_total));
        } else {
        	config[i].fix_position = SIZE_MAX; // should never trigger
        	config[i].fix_step = config[i].step; // but just in case
        }
    }
}
