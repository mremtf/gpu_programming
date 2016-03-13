/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

#ifndef _SCAN_WORKEFFICIENT_KERNEL_H_
#define _SCAN_WORKEFFICIENT_KERNEL_H_

///////////////////////////////////////////////////////////////////////////////
//! Work-efficient compute implementation of scan, one thread per 2 elements
//! Work-efficient: O(log(n)) steps, and O(n) adds.
//! Also shared storage efficient: Uses n elements in shared mem -- no ping-ponging
//! Uses a balanced tree type algorithm.  See Blelloch, 1990 "Prefix Sums
//! and Their Applications", or Prins and Chatterjee PRAM course notes:
//! http://www.cs.unc.edu/~prins/Classes/203/Handouts/pram.pdf
//!
//! Pro: Work Efficient
//! Con: Shared memory bank conflicts due to the addressing used.
//
//! @param g_odata  output data in global memory
//! @param g_idata  input data in global memory
//! @param n        input number of elements to scan from input data
///////////////////////////////////////////////////////////////////////////////
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

#ifndef LOAD_LEVEL
#error "LOAD_LEVEL not defined, something probably went terribly wrong during compilation"
#endif

#define DATA_LOAD_OFFSET (32 >> (LOAD_LEVEL - 1))
#if DATA_LOAD_OFFSET == 0
#error "Load level too high!"
#endif

#define COMPUTATION_LOAD_COUNT ((1 << LOAD_LEVEL) - 1)

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

    for (int i = 0; i < LOAD_LEVEL; ++i, global_in += DATA_LOAD_OFFSET) {
        shared_data[tid] = *global_in;
    }
    __syncthreads();

    shared_data = shared_data + tid;

    for (unsigned int cutoff = blockDim.x >> 1; cutoff > 0; cutoff >>= 1) {
        if (tid < cutoff) {
            // shared_data[tid] += shared_data[tid + cutoff];
            float p_sum          = 0.0f;
            float *shared_offset = shared_data + cutoff;
            for (int i = 0; i < COMPUTATION_LOAD_COUNT; ++i, shared_offset += cutoff) {
                p_sum += *shared_offset;
            }
            *shared_data += p_sum;
        }
        __syncthreads();
    }

    if (tid == 0) {
        // atomically add our block results to the output float
        // shared -> register -> global surely won't be faster than shared->global
        atomicAdd(global_out, *shared_data);
    }
}


#endif  // #ifndef _SCAN_WORKEFFICIENT_KERNEL_H_
