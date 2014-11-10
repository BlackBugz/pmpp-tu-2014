/*
 * convolution_gpu_gmem.h
 *
 *  Created on: 05/nov/2014
 *      Author: marco
 */

#ifndef CONVOLUTION_GPU_CMEM_H_
#define CONVOLUTION_GPU_CMEM_H_

#include "ppm.h"


void ApplyFilterGPUCMem(PPMImage &destImg, PPMImage &srcImg, const float * kernel, unsigned int kernelSize);

#endif /* CONVOLUTION_GPU_CMEM_H_ */
