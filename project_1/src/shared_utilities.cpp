#include <vector>
#include <random>
#include <algorithm>
#include "shared_utilities.hpp"
#include <chrono>
#include <functional>

std::vector<float> generate_vector(size_t n) {
    std::default_random_engine gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dist(0, 100);
    auto get_num = std::bind(dist, std::ref(gen));
    std::vector<float> vec(n);
    std::generate(vec.begin(), vec.end(), get_num);
    return vec;
}

bool check_equal(const std::vector<float> &a, const std::vector<float> &b) {
    return std::equal(a.cbegin(), a.cend(), b.cbegin());
}