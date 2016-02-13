#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <functional>

#include "vector_add.hpp"
#include "parameters.hpp"
#include "device_queries.hpp"

using std::vector;

vector<float> cpu_addition(const vector<float> &a, const vector<float> &b) {
    vector<float> results(a);
    std::transform(results.begin(), results.end(), b.cbegin(), results.begin(), std::plus<float>());
    return a;
}

bool check_equal(const vector<float> &a, const vector<float> &b) {
    return std::equal(a.cbegin(), a.cend(), b.cbegin());
}

std::vector<float> generate_vector(size_t n) {
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(0, 100);
    auto get_num = std::bind(dist, std::ref(gen));
    std::vector<float> vec(n);
    std::generate(vec.begin(), vec.end(), get_num);
    return vec;
}

/*
calculate step, calculate final index, if step doesn't work, specify final index and special step

if special step doesn't exist, FFFFFFFF it. only warp divergence will be in one warp in final block.

ideally, last block on last device
*/

void launch_kernels_and_report(const options_t &opts) {
    return;
}
