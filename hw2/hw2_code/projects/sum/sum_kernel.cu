#ifndef _SCAN_WORKEFFICIENT_KERNEL_H_
#define _SCAN_WORKEFFICIENT_KERNEL_H_

#ifndef LOAD_LEVEL
#error "LOAD_LEVEL not defined, something probably went terribly wrong during compilation"
#endif

#define DATA_READ_OFFSET (32 >> (LOAD_LEVEL - 1))
#if DATA_READ_OFFSET == 0
#error "Load level too high!"
#endif

#define DATA_READ_LOAD (1 << (LOAD_LEVEL - 1))

#define COMPUTATION_LOAD_COUNT ((1 << LOAD_LEVEL) - 1)

__global__ void sum_kernel(float *global_out, float *global_in, const unsigned int n_elem) {
    // finally get to use the reduction code form HPC
    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * 32 /*blockDim.x*/ + threadIdx.x;

    // Writing it such that we always round up to the nearest multiple of the warp size (32)
    // And the extra is set to zero, so we can just do it and not have any issues
    // So ALL threads are working in valid memory
    float *shared_root = sdata + tid;
    float *global_root = global_in + gid;

    for (unsigned int g_block = blockIdx.x; g_block < (n_elem / 32);
         g_block += gridDim.x, global_root += gridDim.x * 32) {
        // if (threadIdx.x == 0) printf("%d DO BLOCK %d OF %d\n", blockIdx.x, g_block, n_elem / 32);

        // Offsets can be calculated of the blockdim, since the blockdim has been set by the load level
        // 32 / blockDim.x is effectively LOAD_LEVEL

        float *global_data = global_root;
        float *shared_data = shared_root;

        for (int i = 0; i < DATA_READ_LOAD; ++i, global_data += DATA_READ_OFFSET, shared_data += DATA_READ_OFFSET) {
            // printf("%d_%d_%d LOAD %f TO %d\n", tid, blockIdx.x, gid, *global_data, shared_data - sdata);
            *shared_data = *global_data;
        }
        // There are no other warps, so no need to sync
        //__syncthreads();

        shared_data = shared_root;

        for (unsigned int i = 0; i < DATA_READ_LOAD; ++i, shared_data += DATA_READ_OFFSET) {
            for (unsigned int cutoff = blockDim.x >> 1; cutoff > 0; cutoff >>= 1) {
                if (tid < cutoff) {
                    // float dbg = *shared_data;
                    // float dbg_2 = shared_data[cutoff];
                    // printf("%d_%d_%d PASS CUTOFF %d\n",tid,blockIdx.x,gid,cutoff);
                    // printf("%d_%d_%d ADDED %d (%f) AND %d
                    // (%f)\n",tid,blockIdx.x,gid,tid+i*DATA_READ_OFFSET,dbg,tid+(i*DATA_READ_OFFSET) + cutoff,dbg_2);
                    *shared_data += shared_data[cutoff];
                }
                //__syncthreads();
            }
            if (i != 0 && tid == 0) {
                // printf("%d_%d_%d SAVE PARTIAL RESULT %f\n",tid,blockIdx.x,gid,*shared_data);
                sdata[0] += *shared_data;
            }
        }

        if (tid == 0) {
            // atomically add our block results to the output float
            // shared -> register -> global surely won't be faster than shared->global
            atomicAdd(global_out, sdata[0]);
        }
    }
}

/*
YOU WERE TOO GOOD FOR THIS WORLD

Basically, it COMPLETELY corrected the work load to a basic reduction after one iteration
But I kinda assumed it would have to be corrected every iteration
So it tried to "help" with the others and inflates the sum

And I can't think of a better way to make the first iteration special without
just putting another if in the loop and that's gross

So I'm just going to dumb it down and make it do the smaller reduction multiple times.

It's slightly less efficient, but I've already made it less efficient by
making the threads adjustable anyway because that's a requirement :/

__global__ void sum_kernel(float *global_out, float *global_in) {
    // finally get to use the reduction code form HPC
    extern __shared__ float sdata[];

    float *shared_data = sdata;

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Writing it such that we always round up to the nearest multiple of the warp size (32)
    // And the extra is set to zero, so we can just do it and not have any issues
    // So ALL threads are working in valid memory

    global_in += gid;

    // Offsets can be calculated of the blockdim, since the blockdim has been set by the load level
    // 32 / blockDim.x is effectively LOAD_LEVEL

    shared_data = sdata + tid;
    for (int i = 0; i < DATA_READ_LOAD; ++i, global_in += DATA_READ_OFFSET, shared_data += DATA_READ_OFFSET) {
printf("%d LOAD %f INTO POSITION %d\n",tid,*global_in,shared_data - sdata);
        *shared_data = *global_in;
    }
    //__syncthreads();

    shared_data = sdata + tid;

    for (unsigned int cutoff = blockDim.x >> 1; cutoff > 0; cutoff >>= 1) {
        if (tid < cutoff) {
printf("TID %d MADE IT TO CUTOFF %d\n",tid,cutoff);
            // shared_data[tid] += shared_data[tid + cutoff];
            float p_sum          = 0.0f;
            float *shared_offset = shared_data + cutoff;
            for (int i = 0; i < COMPUTATION_LOAD_COUNT; ++i, shared_offset += cutoff) {
                p_sum += shared_data[(i+1)*cutoff];
printf("%d ADDED POSITION %d (%f) FOR %f\n",tid,(i+1)*cutoff+tid,shared_data[(i+1)*cutoff],p_sum);
            }
            *shared_data += p_sum;
printf("%d ADD ALL TO BASE %f (+ %f)\n",tid,*shared_data,p_sum);
        }
        //__syncthreads();
    }

    if (tid == 0) {
        // atomically add our block results to the output float
        // shared -> register -> global surely won't be faster than shared->global
        atomicAdd(global_out, *shared_data);
    }
}
*/
/*
No. This is gross, and I can't guarentee it'll work everywhere all the time.
Something just feels WRONG about this idea and I can't put my finger on it.
Thanfully, thought of load level and now I can go back to my nice reduction

__global__ void sum_kernel(float *result, float *data, const unsigned n, const unsigned step,
                           const unsigned fix_position, const unsigned fix_step) {
    extern __shared__ float sdata[];

    const unsigned tid = threadIdx.x;
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        sdata[tid] = data[gid];
        __syncthreads();

        float local_sum     = 0.0f;
        unsigned int offset = tid;

        for (unsigned i = 0; i < step; ++i) {
            local_sum += sdata[tid];
        }

        // what's worse, an n-thread bank conflict, or all threads atomically adding to the same bank?
        // Actually, nevermind, it HAS to be atomic because otherwise it may be dangerous?


        if (tid == 0) {
            // atomically add our block results to the output float
            // shared -> register -> global surely won't be faster than shared->global
            atomicAdd(result, shared_data[0]);
        }
    }
}
*/


#endif  // #ifndef _SCAN_WORKEFFICIENT_KERNEL_H_
