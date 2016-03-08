/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
//#include <test1_kernel.cu>
// function [1 2 3 4 5 6 7 8 9 10]
__device__ float bigfunction1()
{
return (expf(sqrtf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
//function 2 [2 1 3 4 5 6 7 8 9 10]
__device__ float bigfunction2()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 3
__device__ float bigfunction3()
{
return (expf(sqrtf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 4
__device__ float bigfunction4()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 5
__device__ float bigfunction5()
{
return (expf(sqrtf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 6
__device__ float bigfunction6()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 7
__device__ float bigfunction7()
{
return (expf(sqrtf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 8
__device__ float bigfunction8()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 9
__device__ float bigfunction9()
{
return (expf(sqrtf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 10
__device__ float bigfunction10()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 11
__device__ float bigfunction11()
{
return (expf(sqrtf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 12
__device__ float bigfunction12()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 13
__device__ float bigfunction13()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 14
__device__ float bigfunction14()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 16
__device__ float bigfunction15()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 17
__device__ float bigfunction16()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 18
__device__ float bigfunction17()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 19
__device__ float bigfunction18()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 20
__device__ float bigfunction()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 21
__device__ float bigfunctionb()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 22
__device__ float bigfunctionb()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 23
__device__ float bigfunctionb()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 24
__device__ float bigfunctionb()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 25
__device__ float bigfunctionb()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 26
__device__ float bigfunctionb()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 27
__device__ float bigfunctionb()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 28
__device__ float bigfunctionb()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 29
__device__ float bigfunctionb()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 30
__device__ float bigfunctionb()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 31
__device__ float bigfunctionb()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
// function 32
__device__ float bigfunctionb()
{
return (sqrtf(expf(exp2f(exp10f(expm1f(logf(log1pf(sinf(cosf(tanf(float(threadIdx.x))))))))))));
}
////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel template for flops test
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel( float* g_idata, float* g_odata) 
{
    float result=1;
 
    // place variety of branch solutions here
    // make sure you use results, so compiler does not optomize out
    if (threadIdx.x < 128) {
	result = bigfunctiona();
    } else {
	result = bigfunctionb();
    }

     g_odata[0] = result;
}

// handles 8 branches in the code
__global__ void
runEightBranches( float* g_idata, float* g_odata) 
{
    float result=1;
 
    // place variety of branch solutions here
    // make sure you use results, so compiler does not optomize out
    if (threadIdx.x < 128) {
	result = bigfunctiona();
    } else {
	result = bigfunctionb();
    }

     g_odata[0] = result;
}
// handles 16 branches in the code
__global__ void
runSixteenBranches( float* g_idata, float* g_odata) 
{
    float result=1;
 
    // place variety of branch solutions here
    // make sure you use results, so compiler does not optomize out
    if (threadIdx.x < 128) {
	result = bigfunctiona();
    } else {
	result = bigfunctionb();
    }

     g_odata[0] = result;
}
// handles 32 branches in the code
__global__ void
runThirtyTwoBranches( float* g_idata, float* g_odata) 
{
    float result=1;
 
    // place variety of branch solutions here
    // make sure you use results, so compiler does not optomize out
    if (threadIdx.x < 128) {
	result = bigfunctiona();
    } else {
	result = bigfunctionb();
    }

     g_odata[0] = result;
}
// Handles 64 branches of the code
__global__ void
runSixtyFourBranches( float* g_idata, float* g_odata) 
{
    float result=1;
 
    // place variety of branch solutions here
    // make sure you use results, so compiler does not optomize out
    if (threadIdx.x < 128) {
	result = bigfunctiona();
    } else {
	result = bigfunctionb();
    }

     g_odata[0] = result;
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);

    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{

    CUT_DEVICE_INIT();

    unsigned int timer = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    // adjust number of threads here
    unsigned int num_threads = 256;
    unsigned int mem_size = sizeof( float) * num_threads;

    // allocate host memory
    float* h_idata = (float*) malloc( mem_size);
    // initalize the memory
    for( unsigned int i = 0; i < num_threads; ++i) 
    {
        h_idata[i] = (float) i;
    }

    // allocate device memory
    float* d_idata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, mem_size));
    // copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );

    // allocate device memory for result
    float* d_odata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, mem_size));

    // setup execution parameters
    // adjust thread block sizes here
    dim3  grid( 1, 1, 1);
    dim3  threads( num_threads, 1, 1);

    // execute the kernel
    testKernel<<< grid, threads, mem_size >>>( d_idata, d_odata);

    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( mem_size);
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_odata, sizeof( float) * num_threads,
                                cudaMemcpyDeviceToHost) );

    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    CUT_SAFE_CALL( cutDeleteTimer( timer));

    // cleanup memory
    free( h_idata);
    free( h_odata);
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_odata));
}

/*logf( sqrtf( log1pf( tanf( expm1f( exp2f( cosf( expf( sinf( exp10f( 
exp10f( expm1f( log1pf( logf( exp2f( expf( sqrtf( cosf( sinf( tanf( 
expm1f( expf( sinf( exp10f( exp2f( cosf( sqrtf( logf( tanf( log1pf( 
sqrtf( sinf( exp2f( tanf( logf( log1pf( expm1f( expf( exp10f( cosf( 
expf( tanf( sqrtf( logf( exp2f( exp10f( cosf( expm1f( log1pf( sinf( 
tanf( exp2f( sinf( logf( expm1f( exp10f( cosf( sqrtf( log1pf( expf( 
expm1f( expf( cosf( logf( exp10f( tanf( sinf( sqrtf( log1pf( exp2f( 
exp2f( cosf( exp10f( sinf( sqrtf( expf( logf( log1pf( expm1f( tanf( 
logf( sinf( tanf( expf( sqrtf( exp2f( cosf( log1pf( expm1f( exp10f( 
sqrtf( exp2f( expm1f( cosf( logf( log1pf( tanf( expf( exp10f( sinf( 
exp10f( sqrtf( tanf( cosf( expf( logf( exp2f( log1pf( sinf( expm1f( 
logf( tanf( sinf( sqrtf( log1pf( expm1f( exp2f( cosf( expf( exp10f( 
sinf( sqrtf( log1pf( tanf( logf( cosf( exp2f( expm1f( expf( exp10f( 
log1pf( tanf( sqrtf( expm1f( exp2f( exp10f( logf( sinf( cosf( expf( 
sqrtf( exp2f( expm1f( logf( sinf( cosf( exp10f( expf( tanf( log1pf( 
exp2f( expm1f( sqrtf( cosf( logf( expf( sinf( tanf( exp10f( log1pf( 
sinf( sqrtf( log1pf( expf( expm1f( logf( tanf( exp2f( cosf( exp10f( 
logf( sinf( tanf( expf( cosf( exp2f( log1pf( sqrtf( expm1f( exp10f( 
logf( cosf( sqrtf( log1pf( expm1f( expf( sinf( exp2f( exp10f( tanf( 
expm1f( exp2f( log1pf( exp10f( logf( cosf( tanf( sqrtf( sinf( expf( 
sinf( cosf( tanf( exp2f( sqrtf( expm1f( exp10f( logf( expf( log1pf( 
log1pf( sqrtf( cosf( tanf( sinf( exp10f( expf( expm1f( exp2f( logf( 
log1pf( logf( exp10f( expm1f( tanf( expf( sqrtf( sinf( cosf( exp2f( 
exp2f( tanf( expf( cosf( expm1f( logf( sinf( sqrtf( log1pf( exp10f( 
expf( sinf( sqrtf( log1pf( expm1f( exp10f( tanf( logf( exp2f( cosf( 
expm1f( log1pf( expf( cosf( sqrtf( exp10f( logf( exp2f( tanf( sinf( 
log1pf( expf( expm1f( exp2f( sqrtf( tanf( sinf( exp10f( cosf( logf( 
expf( sinf( tanf( log1pf( logf( expm1f( exp2f( sqrtf( cosf( exp10f( 
sqrtf( tanf( log1pf( expf( expm1f( logf( cosf( exp10f( sinf( exp2f( 
expf( sinf( expm1f( sqrtf( tanf( log1pf( logf( cosf( exp10f( exp2f( 
cosf( exp10f( logf( expm1f( log1pf( sinf( tanf( sqrtf( exp2f( expf( 
expf( cosf( logf( log1pf( sinf( sqrtf( expm1f( exp10f( tanf( exp2f( 
exp2f( tanf( logf( sqrtf( cosf( sinf( expf( expm1f( log1pf( exp10f( 
expf( sinf( log1pf( tanf( exp10f( expm1f( sqrtf( logf( exp2f( cosf( 
log1pf( exp10f( sinf( expf( exp2f( tanf( logf( expm1f( sqrtf( cosf( 
expm1f( exp10f( sqrtf( tanf( expf( log1pf( exp2f( logf( sinf( cosf( 
sinf( sqrtf( exp10f( cosf( exp2f( expm1f( expf( tanf( log1pf( logf( 
exp2f( cosf( sinf( tanf( logf( expf( exp10f( sqrtf( log1pf( expm1f( 
log1pf( cosf( exp10f( sqrtf( exp2f( expf( sinf( tanf( logf( expm1f( 
expm1f( sqrtf( log1pf( logf( exp10f( sinf( expf( cosf( exp2f( tanf( 
tanf( expf( exp2f( logf( sqrtf( exp10f( cosf( log1pf( expm1f( sinf( 
exp2f( sinf( tanf( expf( sqrtf( log1pf( exp10f( logf( expm1f( cosf( 
sqrtf( logf( sinf( cosf( exp10f( exp2f( tanf( expf( expm1f( log1pf( 
log1pf( tanf( expf( exp2f( sinf( logf( expm1f( sqrtf( exp10f( cosf( 
sinf( cosf( sqrtf( expf( logf( log1pf( tanf( exp10f( expm1f( exp2f( 
exp10f( sqrtf( logf( expf( expm1f( sinf( cosf( tanf( exp2f( log1pf( 
expm1f( expf( exp10f( sqrtf( cosf( exp2f( log1pf( tanf( logf( sinf( 
exp2f( sqrtf( cosf( tanf( expf( log1pf( exp10f( sinf( expm1f( logf( 
expm1f( exp2f( exp10f( cosf( tanf( sinf( sqrtf( logf( expf( log1pf( 
expm1f( exp10f( sqrtf( cosf( exp2f( log1pf( tanf( logf( sinf( expf( 
cosf( exp10f( logf( sinf( expf( expm1f( tanf( log1pf( exp2f( sqrtf( 
cosf( tanf( sinf( sqrtf( exp2f( logf( exp10f( expf( log1pf( expm1f( 
exp10f( sinf( expm1f( expf( logf( tanf( cosf( log1pf( sqrtf( exp2f( 
logf( tanf( log1pf( exp10f( expf( sinf( expm1f( sqrtf( cosf( exp2f( 
cosf( sqrtf( sinf( expm1f( expf( exp2f( log1pf( tanf( exp10f( logf( 
sinf( expf( exp10f( log1pf( exp2f( logf( tanf( expm1f( sqrtf( cosf( 
expf( logf( exp2f( sqrtf( sinf( cosf( expm1f( tanf( log1pf( exp10f( 
sqrtf( exp10f( exp2f( log1pf( sinf( cosf( logf( expf( tanf( expm1f( 
expf( sqrtf( cosf( expm1f( sinf( log1pf( exp2f( tanf( exp10f( logf( 
exp2f( log1pf( sinf( expm1f( sqrtf( logf( cosf( tanf( exp10f( expf( 
sqrtf( sinf( logf( expf( exp10f( tanf( log1pf( exp2f( expm1f( cosf( 
sinf( cosf( exp2f( expm1f( exp10f( expf( logf( log1pf( sqrtf( tanf( 
expf( sqrtf( exp10f( expm1f( tanf( sinf( logf( exp2f( log1pf( cosf( 
sinf( sqrtf( tanf( expm1f( expf( cosf( log1pf( exp10f( exp2f( logf(*/ 
