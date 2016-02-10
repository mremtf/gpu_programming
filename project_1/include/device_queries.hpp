#ifndef _DEVICE_QUERIES_HPP__
#define _DEVICE_QUERIES_HPP__

#include <cuda_runtime_api.h>
#include <vector>
// Functions to query device count and the global memory of a device

std::vector<int> get_devices() {
    std::vector<int> devices;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == CudaSuccess && device_count) {
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
    if (cudaGetDeviceProperties(&prop_struct, device_id) == CudaSuccess) {
        return prop_struct.totalGlobalMem;
    }
}



#endif