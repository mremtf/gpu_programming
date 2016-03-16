

////////////////////////////////////////////////////////////////////////////////
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>


__global__ void
simple_copy_kernel( float* g_idata, float* g_odata, size_t N) 
{
		// thread copy	
		size_t gtid = blockDim.x * blockIdx.x + threadIdx.x;
		if (gtid < N) 
			g_odata[gtid] = g_idata[gtid];
}



void run_device_mem_local_to_gpu(float* h_idata, size_t h_size, size_t d1, size_t d2) {
    // adjust number of threads here
 		//unsigned int num_threads = h_size;
		// setup execution parameters
    // adjust thread block sizes here
		cudaDeviceReset();
		int grid_size = 0;
		int thread_count = 32;
		if ((h_size % thread_count) != 0) {
			grid_size = (h_size / thread_count + 1) * thread_count;
		}
		else {
			grid_size = h_size / thread_count;
		}

		cudaSetDevice(d1);
    unsigned int mem_size = sizeof( float) * h_size;
		//printf("MEMORY SIZE = %lu", mem_size);

		printf("TOTAL MEM PER MALLOC %lu\n\n", mem_size);
		printf("threads %d block_count %d\n\n", thread_count, grid_size);

    // allocate device memory
    float* d_idata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, mem_size));


    // copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );
    // allocate device memory for result
    float* d_odata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, mem_size));
		
		
    dim3  grid( grid_size, 1, 1);	
    dim3  threads( thread_count, 1, 1);

    // execute the selected kernel
    simple_copy_kernel<<< grid, threads,0>>>( d_idata, d_odata,h_size);
		cudaError_t cuerr = cudaGetLastError() ;
		if( cuerr != cudaSuccess) {
			printf("CUDA ERROR %d\n\n", cuerr);
		}
	
		// check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( mem_size);
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_odata, mem_size,
                                cudaMemcpyDeviceToHost) );


    // cleanup memory
    free( h_odata);
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_odata));

}

void run_remote_peer_to_peer_memory_access(float* h_idata, size_t h_size,size_t d1, size_t d2) {
    // adjust number of threads here
 		//unsigned int num_threads = h_size;
		int grid_size = 0;
		int thread_count = 32;
		if ((h_size % thread_count) != 0) {
			grid_size = (h_size / thread_count + 1) * thread_count;
		}
		else {
			grid_size = h_size / thread_count;
		}
    unsigned int mem_size = sizeof( float) * h_size;

		printf("TOTAL MEM PER MALLOC %lu\n\n", mem_size);
		cudaSetDevice(d1);
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

    dim3  grid( grid_size, 1, 1);
		
    dim3  threads( thread_count, 1, 1);

    // execute the selected kernel
    simple_copy_kernel<<< grid, threads,0>>>( d_idata, d_odata,h_size);

		// check if kernel execution generated and error
 		cudaError_t cuerr = cudaGetLastError(); 
		if( cuerr != cudaSuccess) {
			printf("CUDA ERROR %d\n\n", cuerr);
		}
    CUT_CHECK_ERROR("Kernel execution failed");

		// change GPU device
		cudaSetDevice(d2);

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

		simple_copy_kernel<<< grid, threads,0>>>( d_idata, d_odata_two,h_size);
		cuerr = cudaGetLastError();
		if(  cuerr != cudaSuccess) {
			printf("CUDA ERROR %d\n\n", cuerr);
		}

		// check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

		/*
		* HOST MEMORY COPIES FROM DEVICE ONE
		**/

		cudaSetDevice(d1);
		// allocate mem for the result on host side
    float* h_odata = (float*) malloc( mem_size);
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_odata, mem_size,
                                cudaMemcpyDeviceToHost) );

		/*
		* HOST MEMORY COPIES FROM DEVICE TWO
		**/
		cudaSetDevice(d2);
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
    CUDA_SAFE_CALL(cudaFree(d_odata));
    CUDA_SAFE_CALL(cudaFree(d_odata_two));

}

void run_remote_memory_access_using_data_copy(float* h_idata, size_t h_size, size_t d1, size_t d2) {
    // adjust number of threads here
 		//unsigned int num_threads = h_size;
		int grid_size = 0;
		int thread_count = 32;
		if ((h_size % thread_count) != 0) {
			grid_size = (h_size / thread_count + 1) * thread_count;
		}
		else {
			grid_size = h_size / thread_count;
		}
    
		unsigned int mem_size = sizeof( float) * h_size;
		printf("TOTAL MEM PER MALLOC %lu\n\n", mem_size);
		cudaSetDevice(d1);
		// allocate device memory
    float* d_idata, *d_odata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, mem_size));


    // copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );

		 // allocate device memory for result
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, mem_size));
		//printf ("blocks = %d\n", grid_size);
    dim3  grid( grid_size, 1, 1);	
    dim3  threads( thread_count, 1, 1);

    // execute the selected kernel
    simple_copy_kernel<<< grid, threads,0>>>( d_idata, d_odata, h_size);
		cudaError_t cuerr = cudaGetLastError();
		if(  cuerr != cudaSuccess) {
			printf("CUDA ERROR %d\n\n", cuerr);
			printf("ERROR: %s\n\n",cudaGetErrorString(cuerr));
		}
		// check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");
		// allocate mem for the result on host side
    float* h_odata = (float*) malloc( mem_size);
    
    // copy result from device to host
		CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_odata, mem_size,
                                cudaMemcpyDeviceToHost) );

		// change GPU device
		cudaSetDevice(d2);
				// allocate mem for the result on host side
    float* h_odata_two = (float*) malloc( mem_size);
		// allocate device memory
    float* d_idata_two, *d_odata_two;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata_two, mem_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata_two, mem_size));


		CUDA_SAFE_CALL( cudaMemcpy( d_idata_two, h_odata, mem_size,
                                cudaMemcpyHostToDevice) );


    simple_copy_kernel<<< grid, threads,0>>>( d_idata_two, d_odata_two, h_size);
		cuerr = cudaGetLastError();
		if( cuerr != cudaSuccess) {
			printf("CUDA ERROR %d\n\n", cuerr);
		}

		// check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_odata_two, d_odata_two, mem_size,
                                cudaMemcpyDeviceToHost) );

		if(memcmp(h_odata,h_odata_two,mem_size) != 0) {
			printf("FAILED TO BE EQUAL\n");
		} 

	  // cleanup memory
    free( h_odata);
		free( h_odata_two);
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_idata_two));
    CUDA_SAFE_CALL(cudaFree(d_odata));
    CUDA_SAFE_CALL(cudaFree(d_odata_two));

}

// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
		if (argc != 5) {
			printf("%s <memory access 1 local 2 peer-to-peer 3 peer-to-peer-memcpy> <num elements> <device 1 id> <device 2 id>\n", argv[0]);
			return 0;
		}

    CUT_DEVICE_INIT();
		
		int memory_access = atoi(argv[1]);
		int num_elements = atoi(argv[2]);
		int device_one_id = atoi(argv[3]);
		int device_two_id = atoi(argv[4]);

    // allocate host memory
		if ((num_elements % 32) != 0) {
			num_elements = (num_elements / 32 + 1) * 32; 
		}
		printf("number_elements: %d\n\n", num_elements);

    float* h_idata = (float*) malloc( sizeof(float)* num_elements);
    // initalize the memory
    for( unsigned int i = 0; i < num_elements; ++i) 
    {
        h_idata[i] = 0;
    }

		switch (memory_access) {
			case 1:
				run_device_mem_local_to_gpu(h_idata,num_elements, device_one_id,device_two_id);
			break;
			case 2:	
				run_remote_peer_to_peer_memory_access(h_idata,num_elements,device_one_id,device_two_id);
			break;

			case 3:	
				run_remote_memory_access_using_data_copy(h_idata,num_elements,device_one_id,device_two_id);
			break;
			
			default:
				return -1;	
		}

		free(h_idata);

		return 0;
}
