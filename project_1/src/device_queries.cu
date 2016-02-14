#include "device_queries.hpp"

#include <cuda_runtime_api.h>
#include <stdexcept>
// Functions to query device count and the global memory of a device

std::vector<int> get_devices() {
    std::vector<int> devices;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count) {
        for (int i = 0; i < device_count; ++i) {
        	devices.push_back(i);
        }
        return devices;
    } else {
        throw std::runtime_error("No device and/or cuda freakout!");
    }
}

size_t get_global_mem(const int device_id) {
    cudaDeviceProp prop_struct;
    if (cudaGetDeviceProperties(&prop_struct, device_id) == cudaSuccess) {
        return prop_struct.totalGlobalMem;
    }
    throw std::runtime_error("Bad device id for query!");
}

std::pair<int,int> get_dims(const int device_id) {
    cudaDeviceProp prop_struct;
    if(cudaGetDeviceProperties(&prop_struct, device_id) == cudaSuccess) {
        return std::pair<int,int>(prop_struct.maxThreadsDim[0],prop_struct.maxGridSize[0]);
    }
    throw std::runtime_error("Bad device id for dim query!");
}
