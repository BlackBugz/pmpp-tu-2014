/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <cuda_runtime.h>
#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include "ppm.h"

#define BLOCK_SIZE 32

extern __shared__ float buffer[];

__global__ void ConvolveHGPUSMem(unsigned int *dst, const unsigned int *src, const float *kernel, int kernelSize, int w, int h)
{

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;
	int halfKernel = kernelSize / 2;

	int threadId = ty * blockDim.x + tx;

	float * sharedKernel = buffer;

	// loading the kernel in the shared memory
	if(threadId < kernelSize){
		for(int i = threadId; i < kernelSize; i += blockDim.y+blockDim.x){
			sharedKernel[i] = kernel[i];
		}
	}

	int left = col - MAX(0, col - halfKernel);
	int right = col + MIN(w-1, col + halfKernel);
	int shrW = blockDim.x+2*halfKernel;

	// left and right needs to add more space before and after the bloock to
	// have enough room also for the columns used by the kernel
	unsigned int * sharedSrc = (unsigned int *)&buffer[kernelSize];

	sharedSrc[ty * (2*halfKernel + blockDim.x) + tx+halfKernel] = src[row * w + col];

	// copying the elements needed by the kernel, on the left side of the
	// submatrix we are considering
	if(tx > blockDim.x - left - 1){
		for(int i = 1; i*blockDim.x - tx < left; i ++){
			sharedSrc[ty * shrW +  left + tx - i*blockDim.x] = src[row * w + col-i*blockDim.x];
		}
	}

	if(tx < right){
		for(int i = 1; (i-1)*blockDim.x + tx < right; i++){
			sharedSrc[ty * shrW +  left + tx + i*blockDim.x] = src[row * w + col+i*blockDim.x];
		}
	}

	float finalRed = 0.0f;
	float finalGreen = 0.0f;
	float finalBlue = 0.0f;

	for (int i = 0; i < kernelSize; i++)
	{
		int px = left + tx + (i - halfKernel);

		// Clamp to [0, w-1]
		px = MIN(px, w-1);
		px = MAX(px, 0);

		unsigned int pixel = sharedSrc[ty * shrW + px];

		unsigned char r = pixel & 0x000000ff;
		unsigned char g = (pixel & 0x0000ff00) >> 8;
		unsigned char b = (pixel & 0x00ff0000) >> 16;

		finalRed   += r * sharedKernel[i];
		finalGreen += g * sharedKernel[i];
		finalBlue  += b * sharedKernel[i];
	}

	unsigned char finalRed_uc = roundf(finalRed);
	unsigned char finalGreen_uc = roundf(finalGreen);
	unsigned char finalBlue_uc = roundf(finalBlue);

	unsigned int finalPixel = finalRed_uc
		| (finalGreen_uc << 8)
		| (finalBlue_uc << 16);
	dst[row * w + col] = finalPixel;


}

__global__ void ConvolveVGPUSMem(unsigned int *dst, const unsigned int *src, const float *kernel, int kernelSize, int w, int h){

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	float finalRed = 0.0f;
	float finalGreen = 0.0f;
	float finalBlue = 0.0f;

	int halfKernel = kernelSize / 2;

	int threadId = ty * blockDim.x + tx;

	float * sharedKernel = buffer;

	// loading the kernel in the shared memory
	if(threadId < kernelSize){
		for(int i = threadId; i < kernelSize; i += blockDim.y+blockDim.x){
			sharedKernel[i] = kernel[i];
		}
	}

	int up = row - MAX(0, col - halfKernel);
	int down = row + MIN(h-1, col + halfKernel);

	// left and right needs to add more space before and after the bloock to
	// have enough room also for the columns used by the kernel
	unsigned int * sharedSrc = (unsigned int *)&buffer[kernelSize];

	sharedSrc[(ty + halfKernel) * blockDim.x + tx] = src[row * w + col];

	// copying the elements needed by the kernel, on the left side of the
	// submatrix we are considering
	if(ty > blockDim.y - up - 1){
		for(int i = 1; i*blockDim.y - ty < up; i ++){
			sharedSrc[(ty + halfKernel - i*blockDim.y ) * blockDim.x + tx] = src[(row - i*blockDim.y) * w + col];
		}
	}

	if(ty < down){
		for(int i = 1; (i-1)*blockDim.y + tx < right; i++){
			sharedSrc[(ty + halfKernel + i*blockDim.y) * blockDim.x + tx] = src[(row + i*blockDim.y) * w + col];
		}
	}

	for (int i = 0; i < kernelSize; i++)
	{
		int py = row + (i - kernelSize/2);

		// Clamp to [0, h-1]
		py = MIN(py, h-1);
		py = MAX(py, 0);

		unsigned int pixel = src[py * w + col];

		unsigned char r = pixel & 0x000000ff;
		unsigned char g = (pixel & 0x0000ff00) >> 8;
		unsigned char b = (pixel & 0x00ff0000) >> 16;

		finalRed   += r * kernel[i];
		finalGreen += g * kernel[i];
		finalBlue  += b * kernel[i];
	}

	unsigned char finalRed_uc = roundf(finalRed);
	unsigned char finalGreen_uc = roundf(finalGreen);
	unsigned char finalBlue_uc = roundf(finalBlue);

	unsigned int finalPixel = finalRed_uc
		| (finalGreen_uc << 8)
		| (finalBlue_uc << 16);
	dst[row * w + col] = finalPixel;
}


void ApplyFilterGPUSMem(PPMImage &srcImg, PPMImage &destImg, const float * kernel, unsigned int kernelSize)
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(divUp(srcImg.width,BLOCK_SIZE),divUp(srcImg.height,BLOCK_SIZE));
	int halfKernel = kernelSize/2;
	ConvolveHGPUSMem<<<dimGrid, dimBlock, kernelSize + (srcImg.width + 2*halfKernel)*srcImg.height>>>(destImg.data, srcImg.data, kernel, kernelSize, srcImg.width, srcImg.height);

	  cudaError_t error = cudaGetLastError();
	  if(error != cudaSuccess)
	  {
	    // print the CUDA error message and exit
	    printf("CUDA error: %s\n", cudaGetErrorString(error));
	  }


	srcImg = destImg;

	ConvolveVGPUSMem<<<dimGrid, dimBlock, kernelSize + (srcImg.height + 2*kernelSize)*srcImg.width>>>(destImg.data, srcImg.data, kernel, kernelSize, srcImg.width, srcImg.height);
	  error = cudaGetLastError();
	  if(error != cudaSuccess)
	  {
	    // print the CUDA error message and exit
	    printf("CUDA error: %s\n", cudaGetErrorString(error));
	  }

}
