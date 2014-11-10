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

#include "convolution_gpu_cmem.h"

#define BLOCK_SIZE 32

__constant__ float constKernel[129];


__global__ void ConvolveHGPUCMem(unsigned int *dst, const unsigned int *src, int kernelSize, int w, int h)
{

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	float finalRed = 0.0f;
	float finalGreen = 0.0f;
	float finalBlue = 0.0f;

	for (int i = 0; i < kernelSize; i++)
	{
		int px = col + (i - kernelSize/2);

		// Clamp to [0, w-1]
		px = MIN(px, w-1);
		px = MAX(px, 0);

		unsigned int pixel = src[row * w + px];

		unsigned char r = pixel & 0x000000ff;
		unsigned char g = (pixel & 0x0000ff00) >> 8;
		unsigned char b = (pixel & 0x00ff0000) >> 16;

		finalRed   += r * constKernel[i];
		finalGreen += g * constKernel[i];
		finalBlue  += b * constKernel[i];
	}

	unsigned char finalRed_uc = roundf(finalRed);
	unsigned char finalGreen_uc = roundf(finalGreen);
	unsigned char finalBlue_uc = roundf(finalBlue);

	unsigned int finalPixel = finalRed_uc
		| (finalGreen_uc << 8)
		| (finalBlue_uc << 16);
	dst[row * w + col] = finalPixel;


}

__global__ void ConvolveVGPUCMem(unsigned int *dst, const unsigned int *src, int kernelSize, int w, int h){

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	float finalRed = 0.0f;
	float finalGreen = 0.0f;
	float finalBlue = 0.0f;

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

		finalRed   += r * constKernel[i];
		finalGreen += g * constKernel[i];
		finalBlue  += b * constKernel[i];
	}

	unsigned char finalRed_uc = roundf(finalRed);
	unsigned char finalGreen_uc = roundf(finalGreen);
	unsigned char finalBlue_uc = roundf(finalBlue);

	unsigned int finalPixel = finalRed_uc
		| (finalGreen_uc << 8)
		| (finalBlue_uc << 16);
	dst[row * w + col] = finalPixel;
}


void ApplyFilterGPUCMem(PPMImage &destImg, PPMImage &srcImg, const float  * kernel, unsigned int kernelSize)
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(divUp(srcImg.width,BLOCK_SIZE),divUp(srcImg.height,BLOCK_SIZE));

	CUDA_SUCCEEDED(cudaMemcpyToSymbol(constKernel, kernel, sizeof(float)*kernelSize));

	ConvolveHGPUCMem<<<dimGrid, dimBlock>>>(destImg.data, srcImg.data, kernelSize, srcImg.width, srcImg.height);


	  cudaError_t error = cudaGetLastError();
	  if(error != cudaSuccess)
	  {
	    // print the CUDA error message and exit
	    printf("CUDA error: %s\n", cudaGetErrorString(error));
	    exit(-1);
	  }


	unsigned int * bk;
	bk = srcImg.data;
	srcImg.data = destImg.data;
	destImg.data =bk;
	ConvolveVGPUCMem<<<dimGrid, dimBlock>>>(destImg.data, srcImg.data, kernelSize, srcImg.width, srcImg.height);


	  error = cudaGetLastError();
	  if(error != cudaSuccess)
	  {
	    // print the CUDA error message and exit
	    printf("CUDA error: %s\n", cudaGetErrorString(error));
	    exit(-1);
	  }


	printf("done\n");
}
