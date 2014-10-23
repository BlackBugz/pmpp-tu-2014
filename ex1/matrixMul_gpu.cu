#include <cuda_runtime.h>

#include "common.h"
#include "matrix.h"
#include "matrixMul_gpu.h"

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P);

void
MatrixMulGPU(const Matrix &M, const Matrix &N, Matrix &P)
{
	// TODO Task 4: Determine execution configuration and call CUDA kernel
	dim3 dimBlock(P.width, P.width);
	dim3 dimGrid(1,1);

	MatrixMulKernel<<<dimGrid, dimBlock>>>(M, N, P);
}

// TODO Task 4: Implement matrix multiplication CUDA kernel
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float p = 0;

	for(int i = 0; i < M.height; ++i){
		float m = M.elements[ty * M.pitch + i];
		float n = M.elements[i * N.pitch + tx];
		p += m*n;
	}

	P.elements[ty * P.pitch + tx] =p;
}