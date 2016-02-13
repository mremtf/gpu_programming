#include <vector>
#include <random>
#include <algorithm>
#include "shared_utilities.hpp"

std::vector<float> generate_vector(size_t n) {
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(0, 100);
    auto get_num = std::bind(dist, std::ref(gen));
    std::vector<float> vec(n);
    std::generate(vec.begin(), vec.end(), get_num);
    return vec;
}