#ifndef _PARAMETERS_HPP__
#define _PARAMETERS_HPP__

// Throws all sorts of stuff if we don't get proper params filled out

using options_t = struct {
    bool validate;
    bool multi;
    unsigned blocks, threads;
    double utilization;
};

options_t process_params(int argc, char *argv[]);

#endif