#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>

#include "common.h"
#include "matrix.h"
#include "matrixMul_cpu.h"
#include "matrixMul_gpu.h"
#include "timer.h"

// Matrix sizes (P = M * N)
#define M_WIDTH 1024
#define M_HEIGHT 2048
#define N_WIDTH M_HEIGHT
#define N_HEIGHT M_WIDTH
#define P_WIDTH N_WIDTH
#define P_HEIGHT M_HEIGHT

//----------------------------------------------------------------------------
void
printCUDADevices()
{
	// TODO: Task 2

	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	printf("You have %i devices\n", deviceCount);
	for(int i = 0; i < deviceCount; ++i){
		printf("Prop of device %i:\n",i);
		printf("---------------------\n");

		cudaDeviceProp prop;

		cudaGetDeviceProperties(&prop, i);

		printf("Name:\t%s\n", prop.name);
		printf("Compute capability:\t%i.%i\n", prop.major, prop.minor);
		printf("Multiprocessors count:\t%i\n", prop.multiProcessorCount);
		printf("GPU clock rate:\t%i KHz\n", prop.clockRate);
		printf("Total global memory:\t%lu bytes\n", prop.totalGlobalMem);
		printf("L2 cache size:\t%i bytes\n", prop.l2CacheSize);
		printf("\n");
	}
}

//----------------------------------------------------------------------------
int
main(int argc, char **argv)
{
	printCUDADevices();

	const int height = 100;
	const int width = 75;

	// TODO Task 3: Allocate and initialize CPU matrices
	Matrix M = AllocateMatrixCPU(width, height, true);
	Matrix N = AllocateMatrixCPU(height, width, true);
	Matrix P = AllocateMatrixCPU(height, height, false);

	// TODO Task 5: Start CPU timing

	// TODO Task 3: Run matrix multiplication on the CPU
	MatrixMulCPU(M, N, P);

	unsigned long int sum = 0;
	unsigned long int not0 = 0;

	for(int i =0; i < P.height; ++i){
		for(int j = 0; j < P.width; ++j){
			sum += P.elements[i * P.width + j];
			if(P.elements[i * P.width + j] != 0){
				not0++;
			}
		}
	}

	printf("The sum is %lu with %lu el not 0\n", sum, not0);

	// TODO Task 5: Stop CPU timing and print elapsed time

	// TODO Task 4: Allocate GPU matrices

	// TODO Task 5: Start GPU timing with CUDA events

	// TODO Task 4: Copy CPU matrices to the GPU

	// TODO Task 4: Run matrix multiplication on the GPU

	// TODO Task 4: Copy GPU results to the CPU

	// TODO Task 5: Stop GPU timing with CUDA events and print elapsed time

	// TODO Task 4: Compare CPU and GPU results

	// TODO Task 3/4: Clean up
	FreeMatrixCPU(M);
	FreeMatrixCPU(N);
	FreeMatrixCPU(P);

	return EXIT_SUCCESS;
}

