#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <functional>

#include <cuda_runtime_api.h>

#include "vector_add.hpp"
#include "parameters.hpp"
#include "device_queries.hpp"
#include "shared_utilities.hpp"
#include "timer.hpp"

using std::vector;

vector<float> cpu_addition(const vector<float> &a, const vector<float> &b) {
    vector<float> results(a);
    std::transform(a.begin(), a.end(), b.cbegin(), results.begin(), std::plus<float>());
    return results;
}

/*
calculate step, calculate final index, if step doesn't work, specify final index and special step

if special step doesn't exist, FFFFFFFF it. only warp divergence will be in one warp in final block.

ideally, last block on last device
*/

__global__ void cuda_vector_add(float *a, float *b, int step, int fix_position, int fix_step) {
    unsigned position = blockDim.x * blockIdx.x + threadIdx.x;
    // Interesting thing to test
    // ternary here vs if. Only divergence should be last warp in last block
    // But the ternary will probably slow down everything?
    // It would avoid a warp divergence, though!

    // According to what I read, params are in constant memory
    // so we'll make local copies of the important things?

    // This will explode if it's not registers
    unsigned thread_step = step;
    if (position == fix_position) {
        thread_step = fix_step;
    }
    float *arr_one = a;
    float *arr_two = b;
    arr_one += position;
    arr_two += position;

    for (int i = 0; i < thread_step; ++i, ++arr_one, ++arr_two) {
        *arr_one += *arr_two;
    }
}

using device_config_t = struct {
    int device;
    void *vec_a_device, *vec_b_device;  // vecs gets summed into a
    std::vector<float> a, b, c;
    unsigned step;
    unsigned fix_position;  // UINT_MAX
    unsigned fix_step;
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
        config[i].device         = devices[i];
        config[i].a              = generate_vector(float_vec_size[i]);
        config[i].b              = generate_vector(float_vec_size[i]);
        config[i].c              = std::vector<float>(float_vec_size[i]);
        config[i].step           = float_vec_size[i] / thread_total;
        const bool offset_needed = (config[i].step * thread_total) != float_vec_size[i];
        if (offset_needed) {
            config[i].fix_position = config[i].step * (thread_total - 1);
            config[i].fix_step     = config[i].step + (float_vec_size[i] - (config[i].step * thread_total));
        } else {
            config[i].fix_position = UINT_MAX;        // should never trigger
            config[i].fix_step     = config[i].step;  // but just in case
        }
    }

    std::cout << "Configuration complete, executing across cards." << std::endl;

    // prepare and launch! Woooooo.
    for (int i = 0; i < num_devices; ++i) {
        timer time;
        time.begin();

        if (cudaMalloc(&config[i].vec_a_device, float_vec_size[i] * sizeof(float)) != cudaSuccess
            || cudaMalloc(&config[i].vec_b_device, float_vec_size[i] * sizeof(float)) != cudaSuccess) {
            throw std::runtime_error("Failed to malloc vector!");
        }
        if (cudaMemcpy(config[i].vec_a_device, config[i].a.data(), float_vec_size[i] * sizeof(float),
                       cudaMemcpyHostToDevice)
                != cudaSuccess
            || cudaMemcpy(config[i].vec_b_device, config[i].b.data(), float_vec_size[i] * sizeof(float),
                          cudaMemcpyHostToDevice)
                   != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to device!");
        }

        cuda_vector_add<<<blocks, threads>>>((float *) config[i].vec_a_device, (float *) config[i].vec_b_device,
                                             config[i].step, config[i].fix_position, config[i].fix_step);

        if (cudaMemcpy(config[i].c.data(), config[i].vec_a_device, float_vec_size[i] * sizeof(float),
                       cudaMemcpyDeviceToHost)
            != cudaSuccess) {
            throw std::runtime_error("Could not copy data back! (or kernel launch failed?)");
        }

        cudaFree(config[i].vec_a_device);
        cudaFree(config[i].vec_b_device);

        time.end();

        std::cout << "GPU_" << config[i].device << " time: " << time.ms_elapsed() << " ms" << std::endl;

        if (validate) {
            timer cpu_time;
            cpu_time.begin();
            std::vector<float> cpu_result = cpu_addition(config[i].a, config[i].b);
            cpu_time.end();
            std::cout << "CPU time: " << cpu_time.ms_elapsed() << " ms" << std::endl;
            if (!check_equal(config[i].c, cpu_result)) {
                std::cout << "VERIFICATION FAILED (epsilon issue?)" << std::endl;
                std::cout << config[i].a[0] << " " << config[i].a[1] << " " << config[i].a[2] << std::endl;
                std::cout << config[i].b[0] << " " << config[i].b[1] << " " << config[i].b[2] << std::endl << std::endl;

                std::cout << config[i].c[0] << " " << config[i].c[1] << " " << config[i].c[2] << std::endl;
                std::cout << cpu_result[0] << " " << cpu_result[1] << " " << cpu_result[2] << std::endl;
            }
        }
    }
}
