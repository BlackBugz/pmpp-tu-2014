#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

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
void printCUDADevices()
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
int main(int argc, char **argv)
{
	printCUDADevices();

	const int height = M_HEIGHT;
	const int width = M_WIDTH;

	// TODO Task 3: Allocate and initialize CPU matrices
	Matrix M = AllocateMatrixCPU(width, height, true);
	Matrix N = AllocateMatrixCPU(height, width, true);
	Matrix P = AllocateMatrixCPU(height, height, false);
	Matrix PP = AllocateMatrixCPU(height, height, false);

	// TODO Task 5: Start CPU timing
	double startTime = SysGetTime_ms();

	// TODO Task 3: Run matrix multiplication on the CPU
	MatrixMulCPU(M, N, P);
	checkMatrix(P);

	// TODO Task 5: Stop CPU timing and print elapsed time
	double stopTime = SysGetTime_ms();
	double elapsedTime_CPU = stopTime - startTime;
	printf("CPU processing took: %f ms\n", elapsedTime_CPU);

	// TODO Task 4: Allocate GPU matrices
	Matrix MGPU = AllocateMatrixGPU(width, height);
	Matrix NGPU = AllocateMatrixGPU(height, width);
	Matrix PGPU = AllocateMatrixGPU(height, height);

	// TODO Task 5: Start GPU timing with CUDA events
	cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);

	// TODO Task 4: Copy CPU matrices to the GPU
	CopyToDeviceMatrix(MGPU, M);
	CopyToDeviceMatrix(NGPU, N);

	// TODO Task 4: Run matrix multiplication on the GPU
	MatrixMulGPU(MGPU, NGPU, PGPU);

	// TODO Task 4: Copy GPU results to the CPU
	CopyToHostMatrix(PP, PGPU);
	checkMatrix(PP);

	// TODO Task 5: Stop GPU timing with CUDA events and print elapsed time
	cudaEventRecord(evStop, 0);
	cudaEventSynchronize(evStop);

	float elapsedTime_ms;
	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);

	printf("CUDA processing took: %f ms\n", elapsedTime_ms);
	cudaEventDestroy(evStart);
	cudaEventDestroy(evStop);

	// TODO Task 4: Compare CPU and GPU results
	compareMatrix(PCPU, PGPU);

	// TODO Task 3/4: Clean up
	FreeMatrixCPU(M);
	FreeMatrixCPU(N);
	FreeMatrixCPU(P);

	FreeMatrixGPU(MGPU);
	FreeMatrixGPU(NGPU);
	FreeMatrixGPU(PGPU);

	return EXIT_SUCCESS;
}

