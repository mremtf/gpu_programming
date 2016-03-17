#ifndef LOAD_LEVEL
#define LOAD_LEVEL 1
#endif

#ifndef BLOCK_LOAD_LEVEL
#define BLOCK_LOAD_LEVEL 1
#endif

#define XSTR(x) STR(x)
#define STR(x) #x

#pragma message "LOAD LEVEL = " XSTR(LOAD_LEVEL)
#pragma message "BLOCK LOAD LEVEL = " XSTR(BLOCK_LOAD_LEVEL)

#ifdef _WIN32
#define NOMINMAX
#endif

#define NUM_BANKS 16

// includes, system
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <sum_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

// regression test functionality
extern "C" unsigned int compare(const float *reference, const float *data, const unsigned int len);
extern "C" void computeGold(float *reference, float *idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    runTest(argc, argv);
    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a scan test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
    CUT_DEVICE_INIT();

    int num_elements = 2;
    cutGetCmdLineArgumenti(argc, (const char **) argv, "n", &num_elements);

    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));

    // Nope, 32. Always and forever.
    // Well, not always, but it's determined by the load level
    const unsigned int num_threads = 32 >> (LOAD_LEVEL - 1);
    if (num_threads == 0) {
        // num_threads is not a power of two, which is going to cause... issues. Die.
        printf("Load level too high (%d)! Exiting.", LOAD_LEVEL);
        exit(1);
    }

    // lazy re-adjust because we need it to be a multiple of 32 or we'll explode
    int correct_size = num_elements;
    if (correct_size % 32) {
        correct_size = (correct_size / 32 + 1) * 32;
    }
    unsigned int block_count = (correct_size / 32) >> (BLOCK_LOAD_LEVEL - 1);

    if (block_count == 0) {
        puts("Block count 0! Correcting to 1\n");
        block_count = 1;
    }

    const unsigned int mem_size = sizeof(float) * correct_size;

    float *h_data = (float *) malloc(mem_size);

    // printf("INPUT: ");
    for (unsigned int i = 0; i < num_elements; ++i) {
        h_data[i] = floorf(10 * (rand() / (float) RAND_MAX));
        // printf(" %f ", h_data[i]);
    }
    // blank out extras
    for (unsigned int i = num_elements; i < correct_size; ++i) {
        h_data[i] = 0.0f;
    }
    // printf("\n");

    float *reference = (float *) malloc(mem_size);
    computeGold(reference, h_data, num_elements);

    float *d_idata;
    float *d_odata;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_idata, mem_size));
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_odata, sizeof(float)));

    // sending a literal zero down because I don't want to memset it to 0x00 because
    // what if that is wrong for some reason.
    // AND I can't just have global id 0 blank it because it might get scheduled weird
    // It's dumb but deal with it.
    const float literal_zero = 0.0f;
    // CUDA_SAFE_CALL(cudaMemcpy(d_odata, &literal_zero, sizeof(float), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice));

    CUT_CHECK_ERROR("Kernel execution failed");

    printf("Running sum of %d elements\n", num_elements);

    unsigned int numIterations = 100;

    printf("%d %d\n", block_count, num_threads, correct_size);

    cutStartTimer(timer);
    for (int i = 0; i < numIterations; ++i) {
        CUDA_SAFE_CALL(cudaMemcpy(d_odata, &literal_zero, sizeof(float), cudaMemcpyHostToDevice));
        sum_kernel<<<block_count, num_threads, sizeof(float) * 32>>>(d_odata, d_idata, correct_size);
    }
    cudaThreadSynchronize();
    cutStopTimer(timer);
    printf("Average time: %f ms\n\n", cutGetTimerValue(timer) / numIterations);

    cutResetTimer(timer);

    CUT_CHECK_ERROR("Kernel execution failed");

    CUDA_SAFE_CALL(cudaMemcpy(h_data, d_odata, sizeof(float), cudaMemcpyDeviceToHost));

    printf("OUTPUT: ");
    printf(" %f ", h_data[0]);
    printf("\n");
    printf("REFERENCE: ");
    printf(" %f ", reference[0]);
    printf("\n");

    // custom output handling when no regression test running
    // in this case check if the result is equivalent to the expected soluion

    // We can use an epsilon of 0 since values are integral and in a range
    // that can be exactly represented
    float epsilon               = 0.0f;
    unsigned int result_regtest = cutComparefe(reference, h_data, 1, epsilon);
    printf("sum: Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");

    free(h_data);
    free(reference);
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_odata));
    CUT_SAFE_CALL(cutDeleteTimer(timer));
}
