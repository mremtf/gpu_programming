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

__global__ void cuda_vector_add(float *a, float *b, unsigned step, unsigned total, unsigned fix_position,
                                unsigned fix_step) {
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
        a += position;
        b += position;
        for (int i = 0; i < step; ++i, ++a, ++b) {
            // printf("%p %p %i %i %f %f\n", a, b, position, i, *a, *b);
            *a += *b;
        }
    }
}

using device_config_t = struct {
    int device;
    void *vec_a_device, *vec_b_device;  // vecs gets summed into a
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
    if (!multi) {
        devices.resize(1);
    }
    const size_t num_devices = devices.size();

    std::vector<size_t> float_vec_size(num_devices);
    for (unsigned i = 0; i < num_devices; ++i) {
        float_vec_size[i] = get_global_mem(devices[i]) / sizeof(float) * util / 2.0;
        // number of total floats, get the utilization, div in two because a + b
        // resulting size is the size for vectors a and b
    }

    // Instead of making a giant contiguous vector and serving out slices to the devices
    // I'm just going to make smaller ones since there's no real difference

    std::vector<device_config_t> config(num_devices);
    for (unsigned i = 0; i < num_devices; ++i) {
        auto dim_pair = get_dims(devices[i]);
        if (dim_pair.first < threads || dim_pair.second < blocks) {
            throw std::runtime_error("Block/thread count outside device dims!");
        }
        config[i].device = devices[i];
        // config[i].a      = generate_vector(float_vec_size[i]);
        // config[i].b      = generate_vector(float_vec_size[i]);
        // config[i].c      = std::vector<float>(float_vec_size[i]);
        config[i].step = float_vec_size[i] / thread_total;
        if (config[i].step == 0) {
            std::cout << "More threads than values! Rude!" << std::endl;
            // with a very low mem utilization (read: testing)
            // it will end up with a step of 0 if you get total_threads over n_elem
            // So I guess hardcode 1 and nop anything off the end of the vector
            config[i].step         = 1;
            config[i].fix_position = UINT_MAX;
            config[i].fix_step     = 1;
        } else {
            const bool offset_needed = (config[i].step * thread_total) != float_vec_size[i];
            if (offset_needed) {
                config[i].fix_position = config[i].step * (thread_total - 1);
                config[i].fix_step     = config[i].step + (float_vec_size[i] - (config[i].step * thread_total));
            } else {
                config[i].fix_position = UINT_MAX;        // should never trigger
                config[i].fix_step     = config[i].step;  // but just in case
            }
        }
    }

    std::cout << "Configuration complete, generating data and executing." << std::endl;

    // prepare and launch! Woooooo.
    for (unsigned i = 0; i < num_devices; ++i) {
        timer time;
        /*
        std::cout << "Dev: " << config[i].device << " Step: " << config[i].step << " Fix_P: " << config[i].fix_position
                  << " Fix_s: " << config[i].fix_step << " Threads: " << thread_total
                  << " Val total: " << config[i].a.size() << std::endl;
        */

        std::vector<float> a = generate_vector(float_vec_size[i]);
        std::vector<float> b = generate_vector(float_vec_size[i]);
        std::vector<float> c = std::vector<float>(float_vec_size[i]);

        if (cudaSetDevice(config[i].device) != cudaSuccess) {
            throw std::runtime_error("could not select device!");
        }

        time.begin();

        if (cudaMalloc(&config[i].vec_a_device, float_vec_size[i] * sizeof(float)) != cudaSuccess
            || cudaMalloc(&config[i].vec_b_device, float_vec_size[i] * sizeof(float)) != cudaSuccess) {
            throw std::runtime_error("Failed to malloc vector!");
        }
        if (cudaMemcpy(config[i].vec_a_device, a.data(), float_vec_size[i] * sizeof(float),
                       cudaMemcpyHostToDevice)
                != cudaSuccess
            || cudaMemcpy(config[i].vec_b_device, b.data(), float_vec_size[i] * sizeof(float),
                          cudaMemcpyHostToDevice)
                   != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to device!");
        }

        cuda_vector_add<<<blocks, threads>>>((float *) config[i].vec_a_device, (float *) config[i].vec_b_device,
                                             config[i].step, float_vec_size[i], config[i].fix_position,
                                             config[i].fix_step);

        if (cudaMemcpy(c.data(), config[i].vec_a_device, float_vec_size[i] * sizeof(float),
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
            std::vector<float> cpu_result = cpu_addition(a, b);
            cpu_time.end();
            std::cout << "CPU time: " << cpu_time.ms_elapsed() << " ms" << std::endl;
            if (!check_equal(c, cpu_result)) {
                std::cout << "VERIFICATION FAILED (epsilon issue?)" << std::endl;
                std::cout << a[0] << " " << a[1] << " " << a[2] << std::endl;
                std::cout << b[0] << " " << b[1] << " " << b[2] << std::endl << std::endl;

                std::cout << c[0] << " " << c[1] << " " << c[2] << std::endl;
                std::cout << cpu_result[0] << " " << cpu_result[1] << " " << cpu_result[2] << std::endl;
            }
        }
    }
}
