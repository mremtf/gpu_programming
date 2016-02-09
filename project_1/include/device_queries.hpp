#ifndef _DEVICE_QUERIES_HPP__
#define _DEVICE_QUERIES_HPP__

#include <vector>
// Functions to query device count and the global memory of a device

std::vector<int> get_devices();

unsigned long get_global_mem(int device_id);



#endif