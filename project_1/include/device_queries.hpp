#ifndef _DEVICE_QUERIES_HPP__
#define _DEVICE_QUERIES_HPP__

//#include <cuda_runtime_api.h>
#include <vector>
// Functions to query device count and the global memory of a device

// throws if/when cuda dies for obvious reasons

std::vector<int> get_devices();

size_t get_global_mem(const int device_id);

// returns thread_x_max and grid_x_max
std::pair<int,int> get_dims(const int device_id);

#endif