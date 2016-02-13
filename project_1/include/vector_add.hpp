#ifndef _VECTOR_ADD_HPP__
#define _VECTOR_ADD_HPP__

#include <vector>
#include "parameters.hpp"

using std::vector;

vector<float> cpu_addition(const vector<float> &a, const vector<float> &b);

bool check_equal(const vector<float> &a, const vector<float> &b);

void launch_kernels_and_report(const options_t &opts);

#endif