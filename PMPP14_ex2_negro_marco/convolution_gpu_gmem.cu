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

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__host__ __device__ unsigned int bitreverse(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

__global__ void ConvolveHGPUGMem(unsigned int *dst, const unsigned int *src, const float *kernel, int kernelSize, int w, int h);
__global__ void ConvolveVGPUGMem(unsigned int *dst, const unsigned int *src, const float *kernel, int kernelSize, int w, int h);

void ApplyFilterGPUGMem(PPMImage &srcImg, PPMImage &destImg, const float * kernel, unsigned int kernelSize)
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(divUp(srcImg.width,BLOCK_SIZE),divUp(srcImg.height,BLOCK_SIZE));

	ConvolveHGPUGMem<<<dimGrid, dimBlock>>>(destImg.data, srcImg.data, kernel, kernelSize, srcImg.width, srcImg.height);

	srcImg = destImg;

	ConvolveVGPUGMem<<<dimGrid, dimBlock>>>(destImg.data, srcImg.data, kernel, kernelSize, srcImg.width, srcImg.height);
}

__global__ void ConvolveHGPUGMem(unsigned int *dst, const unsigned int *src, const float *kernel, int kernelSize, int w, int h)
{


	float finalRed = 0.0f;
	float finalGreen = 0.0f;
	float finalBlue = 0.0f;

	for (int i = 0; i < kernelSize; i++)
	{
		int px = x + (i - kernelSize/2);

		// Clamp to [0, w-1]
		px = std::min(px, w-1);
		px = std::max(px, 0);

		unsigned int pixel = src[y * w + px];

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
	dst[y * w + x] = finalPixel;


}

__global__ void ConvolveVGPUGMem(unsigned int *dst, const unsigned int *src, const float *kernel, int kernelSize, int w, int h){

}
