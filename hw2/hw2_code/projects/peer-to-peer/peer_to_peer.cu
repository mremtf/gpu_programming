

////////////////////////////////////////////////////////////////////////////////
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>


__global__ void
simple_copy_kernel( float* g_idata, float* g_odata) 
{
		// thread copy	
		g_odata[threadIdx.x] = g_idata[threadIdx.x];
}



void run_device_mem_local_to_gpu(float* h_idata, size_t h_size) {
    // adjust number of threads here
 		//unsigned int num_threads = h_size;
    unsigned int mem_size = sizeof( float) * h_size;

		unsigned int timer = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
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
    dim3  threads( 32, 1, 1);

    // execute the selected kernel
    simple_copy_kernel<<< grid, threads, mem_size >>>( d_idata, d_odata);
	
		// check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( mem_size);
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_odata, mem_size,
                                cudaMemcpyDeviceToHost) );

    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    CUT_SAFE_CALL( cutDeleteTimer( timer));

    // cleanup memory
    free( h_odata);
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_odata));

}

void run_remote_peer_to_peer_memory_access(float* h_idata, size_t h_size) {
    // adjust number of threads here
 		//unsigned int num_threads = h_size;
    unsigned int mem_size = sizeof( float) * h_size;

		cudaSetDevice(0);
		// allocate device memory
		/*
		* DEVICE MEMORY ALLOCATIONS FOR DEVICE ONE
		*/
    float* d_idata, *d_odata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, mem_size));

    // allocate device memory for result
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, mem_size));

		/*
		* MEMORY COPIES FOR DEVICE ONE
		**/

		// copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );

		// setup execution parameters
    // adjust thread block sizes here
    dim3  grid( 1, 1, 1);
    dim3  threads( 32, 1, 1);

    // execute the selected kernel
    simple_copy_kernel<<< grid, threads, mem_size >>>( d_idata, d_odata);

		// change GPU device
		cudaSetDevice(1);

		/*
		* DEVICE MEMORY ALLOCATIONS FOR DEVICE TWO
		**/

		// allocate device memory
    float* d_odata_two;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata_two, mem_size));

		// Allow access to data between cards
		cudaDeviceEnablePeerAccess(0,0);
		// execute the selected kernel
		//CUDA_SAFE_CALL(cudaMemcpyPeer(d_idata_two,1,d_idata,0,mem_size));

    // execute the selected kernel
    simple_copy_kernel<<< grid, threads, mem_size >>>( d_idata, d_odata_two);

		/*
		* HOST MEMORY COPIES FROM DEVICE ONE
		**/

		cudaSetDevice(0);
		// allocate mem for the result on host side
    float* h_odata = (float*) malloc( mem_size);
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_odata, mem_size,
                                cudaMemcpyDeviceToHost) );

		/*
		* HOST MEMORY COPIES FROM DEVICE TWO
		**/
		cudaSetDevice(1);
		// allocate mem for the result on host side
    float* h_odata_two = (float*) malloc( mem_size);
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_odata_two, d_odata_two, mem_size,
                                cudaMemcpyDeviceToHost) );

		if(memcmp(h_odata,h_odata_two,mem_size) != 0) {
			printf("FAILED TO BE EQUAL\n");
		} 

		/*
		* MEMORY CLEAN UP
		**/
	  // cleanup memory
    free( h_odata);
		free( h_odata_two);
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_odata_two));
}

void run_remote_memory_access_using_data_copy(float* h_idata, size_t h_size) {
    // adjust number of threads here
 		//unsigned int num_threads = h_size;
    unsigned int mem_size = sizeof( float) * h_size;

		// copy to GPU 
		
		// copy back from GPU
		// copy to another GPU

		cudaSetDevice(0);
		// allocate device memory
    float* d_idata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, mem_size));
    // copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );
		// allocate mem for the result on host side
    float* h_odata = (float*) malloc( mem_size);
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_idata, mem_size,
                                cudaMemcpyDeviceToHost) );
		// change GPU device
		cudaSetDevice(1);
		// allocate device memory
    float* d_idata_two;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata_two, mem_size));
		CUDA_SAFE_CALL( cudaMemcpy( d_idata_two, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );
		// allocate mem for the result on host side
    float* h_odata_two = (float*) malloc( mem_size);
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_odata_two, d_idata_two, mem_size,
                                cudaMemcpyDeviceToHost) );

		if(memcmp(h_odata,h_odata_two,mem_size) != 0) {
			printf("FAILED TO BE EQUAL\n");
		} 

	  // cleanup memory
    free( h_odata);
		free( h_odata_two);
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_idata_two));

}

// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
		if (argc != 3) {
			printf("%s <memory access 1 local 2 peer-to-peer 3 peer-to-peer-memcpy> <num elements\n", argv[0]);
			return 0;
		}

    CUT_DEVICE_INIT();
		
		int memory_access = atoi(argv[1]);
		int num_elements = atoi(argv[2]);

    // allocate host memory
    float* h_idata = (float*) malloc( sizeof(float)* num_elements);
    // initalize the memory
    for( unsigned int i = 0; i < num_elements; ++i) 
    {
        h_idata[i] = (float) i;
    }

		switch (memory_access) {
			case 1:
				run_device_mem_local_to_gpu(h_idata,num_elements);
			break;
			case 2:	
				run_remote_peer_to_peer_memory_access(h_idata,num_elements);
			break;

			case 3:	
				run_remote_memory_access_using_data_copy(h_idata,num_elements);
			break;
			
			default:
				return -1;	
		}

		free(h_idata);

		return 0;
}
