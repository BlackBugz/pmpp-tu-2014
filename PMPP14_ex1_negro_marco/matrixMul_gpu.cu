#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include "matrix.h"
#include "matrixMul_gpu.h"

#define BLOCK_SIZE 32

// -------------------------------------------------------------------------------
// Simple implementation, no memory enhancements

__global__ void MatrixMulKernelSimple(Matrix M, Matrix N, Matrix P);

void MatrixMulGPUSimple(const Matrix &M, const Matrix &N, Matrix &P)
{
	// TODO Task 4: Determine execution configuration and call CUDA kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(divUp(N.width,BLOCK_SIZE),divUp(M.height,BLOCK_SIZE));

	MatrixMulKernelSimple<<<dimGrid, dimBlock>>>(M, N, P);
}

// TODO Task 4: Implement matrix multiplication CUDA kernel

/** \brief Kernel function to perform multiplication on the device
 *  \param M First input matrix
 *  \param N Second input matrix
 *  \param P Output matrix
 */
__global__ void MatrixMulKernelSimple(Matrix M, Matrix N, Matrix P)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float p = 0;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	for(int i = 0; i < M.width; ++i){
		float m = M.elements[row * M.pitch/sizeof(float) + i];
		float n = N.elements[i * N.pitch/sizeof(float) + col];
		p += m*n;
	}

	P.elements[row * P.pitch/sizeof(float) + col] = p;
}



// -----------------------------------------------------------------------------------
// Tiled multiplication


__device__ float GetElement(const Matrix A, int row, int col)
{
	return A.elements[row * A.pitch/sizeof(float) + col];
}

__device__ void SetElement(Matrix A, int row, int col, float val)
{
	A.elements[row * A.pitch/sizeof(float) + col] = val;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
	Matrix Msub;
	Msub.width = BLOCK_SIZE;
	Msub.height = BLOCK_SIZE;
	Msub.pitch = A.pitch;
	Msub.elements = &A.elements[A.pitch/sizeof(float) * BLOCK_SIZE * row + BLOCK_SIZE * col];

	return Msub;
}

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P);

void MatrixMulGPU(const Matrix &M, const Matrix &N, Matrix &P)
{
	// TODO Task 4: Determine execution configuration and call CUDA kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(divUp(N.width,dimBlock.x),divUp(M.height,dimBlock.y));

	MatrixMulKernel<<<dimGrid, dimBlock>>>(M, N, P);
}

// TODO Task 4: Implement matrix multiplication CUDA kernel

/** \brief Kernel function to perform multiplication on the device
 *  \param M First input matrix
 *  \param N Second input matrix
 *  \param P Output matrix
 */
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// getting the submatrix for this block
	Matrix SubP = GetSubMatrix(P, bx, by);

	float p = 0;

	for(int i = 0; i < (M.width/BLOCK_SIZE); ++i){

		// Getting the input sub-matrices
		Matrix SubM = GetSubMatrix(M, i, by);
		Matrix SubN = GetSubMatrix(N, bx, i);

		// copying in shared mem
		__shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

		Ms[ty][tx] = GetElement(SubM, tx, ty);
		Ns[ty][tx] = GetElement(SubN, tx, ty);

		__syncthreads();

		for(int k = 0; k < BLOCK_SIZE; ++k)
			p += Ms[ty][k] * Ns[k][tx];


		__syncthreads();
	}

	SetElement(SubP, tx, ty, p);
}
