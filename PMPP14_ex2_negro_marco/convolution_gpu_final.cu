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

__constant__ float constKernelFinal[129];

texture<unsigned int, cudaTextureType1D, cudaReadModeElementType> texRefImg;

__global__ void ConvolveHGPUFinal(unsigned int *dst, const float * kernel, int kernelSize, int w, int h)
{
	extern __shared__ unsigned int buffer[];
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;
	int halfKernel = kernelSize / 2;

	int left = col - MAX(0, col - halfKernel);
	int right = col + MIN(w-1, col + halfKernel);
	int shrW = blockDim.x+2*halfKernel;

	// left and right needs to add more space before and after the bloock to
	// have enough room also for the columns used by the kernel
	unsigned int * sharedSrc = buffer;

	sharedSrc[ty * (2*halfKernel + blockDim.x) + tx+halfKernel] = tex1Dfetch(texRefImg,row * w + col);

	// copying the elements needed by the kernel, on the left side of the
	// submatrix we are considering
	if(tx > blockDim.x - left - 1){
		for(int i = 1; i*blockDim.x - tx < left; i ++){
			sharedSrc[ty * shrW +  left + tx - i*blockDim.x] = tex1Dfetch(texRefImg,row * w + col-i*blockDim.x);
		}
	}

	if(tx < right){
		for(int i = 1; (i-1)*blockDim.x + tx < right; i++){
			sharedSrc[ty * shrW +  left + tx + i*blockDim.x] = tex1Dfetch(texRefImg,row * w + col+i*blockDim.x);
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

__global__ void ConvolveVGPUFinal(unsigned int *dst, int kernelSize, int w, int h){

	extern __shared__ unsigned int buffer[];
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

	int up = row - MAX(0, col - halfKernel);
	int down = row + MIN(h-1, col + halfKernel);

	// left and right needs to add more space before and after the bloock to
	// have enough room also for the columns used by the kernel
	unsigned int * sharedSrc = buffer;

	sharedSrc[(ty + halfKernel) * blockDim.x + tx] = tex1Dfetch(texRefImg,row * w + col);

	// copying the elements needed by the kernel, on the left side of the
	// submatrix we are considering
	if(ty > blockDim.y - up - 1){
		for(int i = 1; i*blockDim.y - ty < up; i ++){
			sharedSrc[(ty + halfKernel - i*blockDim.y ) * blockDim.x + tx] = tex1Dfetch(texRefImg,(row - i*blockDim.y) * w + col);
		}
	}

	if(ty < down){
		for(int i = 1; (i-1)*blockDim.y + ty < down; i++){
			sharedSrc[(ty + halfKernel + i*blockDim.y) * blockDim.x + tx] = tex1Dfetch(texRefImg,(row + i*blockDim.y) * w + col);
		}
	}

	for (int i = 0; i < kernelSize; i++)
	{
		int py = row + (i - kernelSize/2);

		// Clamp to [0, h-1]
		py = MIN(py, h-1);
		py = MAX(py, 0);

		unsigned int pixel = tex1Dfetch(texRefImg,py * w + col);

		unsigned char r = pixel & 0x000000ff;
		unsigned char g = (pixel & 0x0000ff00) >> 8;
		unsigned char b = (pixel & 0x00ff0000) >> 16;

		finalRed   += r * constKernelFinal[i];
		finalGreen += g * constKernelFinal[i];
		finalBlue  += b * constKernelFinal[i];
	}

	unsigned char finalRed_uc = roundf(finalRed);
	unsigned char finalGreen_uc = roundf(finalGreen);
	unsigned char finalBlue_uc = roundf(finalBlue);

	unsigned int finalPixel = finalRed_uc
			| (finalGreen_uc << 8)
			| (finalBlue_uc << 16);
	dst[row * w + col] = finalPixel;
}


void ApplyFilterGPUFinal(PPMImage &srcImg, PPMImage &destImg, const float * kernel, unsigned int kernelSize)
{
	CUDA_SUCCEEDED(cudaBindTexture(0, texRefImg, srcImg.data, srcImg.height*srcImg.width*sizeof(unsigned int)));
	printf("bind done\n");
	CUDA_SUCCEEDED(cudaMemcpyToSymbol(constKernelFinal, kernel, sizeof(float)*kernelSize));
	printf("init 1 done\n");
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(divUp(srcImg.width,BLOCK_SIZE),divUp(srcImg.height,BLOCK_SIZE));
	int halfKernel = kernelSize/2;
	ConvolveHGPUFinal<<<dimGrid, dimBlock, ((srcImg.width + 2*halfKernel)*srcImg.height)>>>(destImg.data, kernel, kernelSize, srcImg.width, srcImg.height);

	printf("comp 1 done\n");
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error at  %s:%i : (%i) %s\n",
				__FILE__, __LINE__, error, cudaGetErrorString(error));
	}

	error = cudaDeviceSynchronize();
	if( cudaSuccess != error )
	{
		fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : (%i) %s\n",
				__FILE__, __LINE__, error, cudaGetErrorString( error ) );
	}

	CUDA_SUCCEEDED(cudaUnbindTexture(texRefImg));

	unsigned int * bk;
	bk = srcImg.data;
	srcImg.data = destImg.data;
	destImg.data =bk;

	CUDA_SUCCEEDED(cudaBindTexture(0, texRefImg, srcImg.data, destImg.height*destImg.width*sizeof(unsigned int)));

	printf("bind 2 done\n");
	ConvolveVGPUFinal<<<dimGrid, dimBlock, ((srcImg.height + 2*kernelSize)*srcImg.width)>>>(destImg.data, kernelSize, srcImg.width, srcImg.height);
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error at  %s:%i : %s\n",
						__FILE__, __LINE__, cudaGetErrorString(error));
	}

	error = cudaDeviceSynchronize();
	if( cudaSuccess != error )
	{
		fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
				__FILE__, __LINE__, cudaGetErrorString( error ) );
	}
	CUDA_SUCCEEDED(cudaUnbindTexture(texRefImg));
	printf("compute 2 done\n");
}
