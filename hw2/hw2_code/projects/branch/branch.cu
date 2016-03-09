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

// BIG DEVICE FUNCTION 
__device__ float bigfunction1(){
return ( exp2f( cosf( exp10f( log1pf( expm1f( logf( tanf( sqrtf( expf( sinf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction2(){
return (sinf( logf( cosf( log1pf( exp2f( tanf( exp10f( expf( sqrtf( expm1f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction3(){
return (exp2f( exp10f( expf( logf( cosf( log1pf( sinf( sqrtf( expm1f( tanf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction4(){
return (exp10f( expm1f( exp2f( sinf( tanf( cosf( expf( sqrtf( log1pf( logf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction5(){
return (sqrtf( log1pf( logf( cosf( expf( sinf( exp10f( expm1f( tanf( exp2f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction6(){
return (sinf( expf( sqrtf( expm1f( exp10f( cosf( logf( log1pf( tanf( exp2f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction7(){
return (logf( exp2f( exp10f( expm1f( expf( cosf( log1pf( sqrtf( sinf( tanf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction8(){
return (sinf( log1pf( exp2f( expf( logf( tanf( expm1f( sqrtf( exp10f( cosf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction9(){
return (logf( tanf( sinf( exp2f( log1pf( cosf( exp10f( sqrtf( expm1f( expf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction10(){
return (expf( cosf( exp2f( exp10f( expm1f( sinf( log1pf( tanf( logf( sqrtf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction11(){
return (exp2f( logf( expf( sqrtf( sinf( expm1f( cosf( tanf( exp10f( log1pf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction12(){
return (logf( cosf( exp2f( expm1f( sinf( exp10f( expf( log1pf( tanf( sqrtf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction13(){
return (sqrtf( logf( exp10f( expm1f( exp2f( tanf( log1pf( sinf( expf( cosf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction14(){
return (log1pf( tanf( sinf( expm1f( logf( exp10f( cosf( exp2f( sqrtf( expf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction15(){
return (sinf( tanf( logf( exp2f( cosf( expf( exp10f( sqrtf( log1pf( expm1f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction16(){
return (cosf( exp2f( expm1f( sqrtf( expf( exp10f( tanf( sinf( logf( log1pf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction17(){
return (cosf( tanf( log1pf( expf( exp10f( exp2f( sinf( sqrtf( logf( expm1f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction18(){
return (sqrtf( sinf( expm1f( tanf( log1pf( cosf( expf( exp2f( exp10f( logf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction19(){
return (cosf( tanf( exp10f( exp2f( expf( expm1f( sinf( logf( log1pf( sqrtf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction20(){
return (sinf( exp2f( logf( expf( sqrtf( tanf( exp10f( expm1f( log1pf( cosf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction21(){
return (expf( log1pf( sinf( tanf( cosf( logf( sqrtf( expm1f( exp10f( exp2f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction22(){
return (expf( tanf( sqrtf( exp10f( exp2f( expm1f( sinf( logf( log1pf( cosf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction23(){
return (exp10f( sinf( tanf( log1pf( cosf( sqrtf( exp2f( logf( expm1f( expf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction24(){
return (logf( exp2f( expf( cosf( sinf( expm1f( sqrtf( log1pf( tanf( exp10f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction25(){
return (expf( exp2f( expm1f( exp10f( log1pf( logf( tanf( sinf( cosf( sqrtf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction26(){
return (expf( cosf( sqrtf( tanf( exp10f( sinf( log1pf( logf( expm1f( exp2f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction27(){
return (
exp2f( expf( log1pf( expm1f( exp10f( cosf( logf( sqrtf( sinf( tanf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction28(){
return (sinf( expf( expm1f( tanf( sqrtf( exp2f( cosf( log1pf( logf( exp10f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction29(){
return (exp10f( expf( logf( expm1f( log1pf( sqrtf( sinf( cosf( tanf( exp2f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction30(){
return (exp2f( cosf( expm1f( exp10f( sqrtf( log1pf( expf( sinf( tanf( logf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction31(){
return (sqrtf( expm1f( exp10f( expf( cosf( tanf( exp2f( sinf( log1pf( logf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction32(){
return (log1pf( expf( logf( sqrtf( exp2f( tanf( sinf( cosf( exp10f( expm1f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction33(){
return (exp2f( logf( sqrtf( expf( exp10f( tanf( cosf( log1pf( expm1f( sinf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction34(){
return (sqrtf( log1pf( exp2f( expm1f( sinf( cosf( logf( tanf( exp10f( expf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction35(){
return (log1pf( sqrtf( exp2f( logf( expm1f( cosf( sinf( tanf( expf( exp10f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction36(){
return (log1pf( exp2f( exp10f( sinf( tanf( sqrtf( logf( expf( expm1f( cosf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction37(){
return (expm1f( expf( cosf( exp2f( tanf( log1pf( exp10f( logf( sqrtf( sinf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction38(){
return (exp2f( logf( expf( expm1f( exp10f( log1pf( tanf( cosf( sinf( sqrtf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction39(){
return (exp2f( log1pf( sqrtf( expf( logf( sinf( exp10f( cosf( expm1f( tanf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction40(){
return (exp2f( log1pf( sqrtf( cosf( sinf( expm1f( tanf( logf( exp10f( expf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction41(){
return (exp2f( expm1f( sinf( cosf( tanf( logf( expf( sqrtf( exp10f( log1pf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction42(){
return (exp10f( sqrtf( sinf( cosf( log1pf( logf( expf( expm1f( tanf( exp2f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction43(){
return (sinf( expf( exp2f( logf( tanf( log1pf( expm1f( sqrtf( exp10f( cosf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction44(){
return (log1pf( sqrtf( tanf( exp2f( sinf( exp10f( expf( logf( expm1f( cosf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction45(){
return (log1pf( cosf( tanf( sinf( logf( exp10f( expm1f( expf( sqrtf( exp2f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction46(){
return (tanf( log1pf( cosf( expf( logf( exp10f( expm1f( exp2f( sinf( sqrtf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction47(){
return (expf( sinf( cosf( exp2f( expm1f( log1pf( exp10f( sqrtf( logf( tanf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction48(){
return (expf( exp10f( sqrtf( logf( expm1f( sinf( exp2f( cosf( log1pf( tanf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction49(){
return (cosf( exp10f( logf( expm1f( expf( sqrtf( tanf( log1pf( sinf( exp2f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction50(){
return (log1pf( expm1f( exp10f( sinf( exp2f( expf( cosf( logf( sqrtf( tanf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction51(){
return (exp2f( sqrtf( logf( cosf( log1pf( expf( exp10f( expm1f( sinf( tanf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction52(){
return (tanf( expm1f( logf( sqrtf( expf( sinf( cosf( log1pf( exp10f( exp2f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction53(){
return (cosf( sinf( expf( expm1f( log1pf( exp2f( tanf( logf( exp10f( sqrtf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction54(){
return (log1pf( sqrtf( expf( sinf( exp2f( cosf( logf( expm1f( tanf( exp10f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction55(){
return (sinf( sqrtf( cosf( exp2f( tanf( logf( log1pf( expf( exp10f( expm1f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction56(){
return (cosf( sinf( exp10f( sqrtf( expf( tanf( exp2f( log1pf( logf( expm1f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction57(){
return (cosf( logf( exp10f( sqrtf( expm1f( log1pf( tanf( exp2f( expf( sinf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction58(){
return (sqrtf( log1pf( exp10f( cosf( expm1f( sinf( exp2f( logf( expf( tanf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction59(){
return (log1pf( expm1f( tanf( logf( expf( exp10f( exp2f( sinf( cosf( sqrtf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction60(){
return (expm1f( expf( cosf( log1pf( logf( exp10f( tanf( sinf( exp2f( sqrtf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction61(){
return (expm1f( expf( sqrtf( sinf( exp2f( logf( exp10f( cosf( log1pf( tanf( float(threadIdx.x))))))))))));
}
__device__ float bigfunction62(){
return (logf( exp2f( sinf( tanf( cosf( log1pf( sqrtf( expf( expm1f( exp10f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction63(){
return (sinf( tanf( exp10f( expf( cosf( logf( log1pf( exp2f( sqrtf( expm1f( float(threadIdx.x))))))))))));
}
__device__ float bigfunction64(){
return (cosf( sinf( expf( expm1f( exp2f( tanf( sqrtf( exp10f( logf( log1pf( float(threadIdx.x))))))))))));
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

// handles 1 branches in the code
__global__ void
runOnesBranches( float* g_idata, float* g_odata) 
{
    float result=1;
 
    // place variety of branch solutions here
    // make sure you use results, so compiler does not optomize out
    if (threadIdx.x == 0) {
			result = bigfunction1();
    } else {
			result = bigfunction2();
    }
}


// handles 2 branches in the code
__global__ void
runTwoBranches( float* g_idata, float* g_odata) 
{
    float result=1;
 
    // place variety of branch solutions here
    // make sure you use results, so compiler does not optomize out
    switch(threadIdx.x) {
			case 0:
				result = bigfunction1();
			break;
    	case 1:
				result = bigfunction2();
    	break;

			default:
				result = bigfunction3();
		}

		g_odata[threadIdx.x] = result;
}

// handles 4 branches in the code
__global__ void
runFourBranches( float* g_idata, float* g_odata) 
{
    float result=1;
 
    // place variety of branch solutions here
    // make sure you use results, so compiler does not optomize out
    switch(threadIdx.x) {
			case 0:
				result = bigfunction1();
			break;

    	case 1:
				result = bigfunction2();
    	break;

    	case 2:
				result = bigfunction2();
    	break;

			default:
				result = bigfunction3();
		}

		g_odata[threadIdx.x] = result;

}


// handles 8 branches in the code
__global__ void
runEightBranches( float* g_idata, float* g_odata) 
{
    float result=1;
 
    // place variety of branch solutions here
    // make sure you use results, so compiler does not optomize out
    switch(threadIdx.x) {
			case 0:
				result = bigfunction1();
			break;

    	case 1:
				result = bigfunction2();
    	break;

    	case 2:
				result = bigfunction3();
    	break;

    	case 3:
				result = bigfunction4();
    	break;

    	case 4:
				result = bigfunction5();
    	break;

    	case 5:
				result = bigfunction6();
    	break;

			case 6:
				result = bigfunction7();
    	break;

			case 7:
				result = bigfunction8();
    	break;

			default:
				result = bigfunction9();
		}

		g_odata[threadIdx.x] = result;
}
// handles 16 branches in the code
__global__ void
runSixteenBranches( float* g_idata, float* g_odata) 
{
    float result=1;
 
    // place variety of branch solutions here
    // make sure you use results, so compiler does not optomize out
    switch(threadIdx.x) {
			case 0:
				result = bigfunction1();
			break;

    	case 1:
				result = bigfunction2();
    	break;

    	case 2:
				result = bigfunction3();
    	break;

    	case 3:
				result = bigfunction4();
    	break;

    	case 4:
				result = bigfunction5();
    	break;

    	case 5:
				result = bigfunction6();
    	break;

			case 6:
				result = bigfunction7();
    	break;

			case 7:
				result = bigfunction8();
    	break;

			case 8:
				result = bigfunction9();
			break;

    	case 9:
				result = bigfunction10();
    	break;

    	case 10:
				result = bigfunction11();
    	break;

    	case 11:
				result = bigfunction12();
    	break;

    	case 12:
				result = bigfunction13();
    	break;

    	case 13:
				result = bigfunction14();
    	break;

			case 14:
				result = bigfunction15();
    	break;

			case 15:
				result = bigfunction16();
    	break;

			default:
				result = bigfunction17();
		}

		g_odata[threadIdx.x] = result;
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

		int num_branches = atoi(argv[1]);

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
		switch (num_branches) {
			case 1:
    		runOneBranches<<< grid, threads, mem_size >>>( d_idata, d_odata);
			break;
			case 2:
    		runTwoBranchestest<<< grid, threads, mem_size >>>( d_idata, d_odata);
			break;
			case 4:
    		runFourBranches<<< grid, threads, mem_size >>>( d_idata, d_odata);
			break;
			case 8:
    		runEightBranches<<< grid, threads, mem_size >>>( d_idata, d_odata);
			break;
			case 16:
    		runSixteenBranches<<< grid, threads, mem_size >>>( d_idata, d_odata);
			break;
			case 32:
    		runThirtyTwoBranches<<< grid, threads, mem_size >>>( d_idata, d_odata);
			break;
			case 64:
    		runSixtyFourBranches<<< grid, threads, mem_size >>>( d_idata, d_odata);
			break;
			default:
    		runNoBranches<<< grid, threads, mem_size >>>( d_idata, d_odata);
			break;
		}

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
