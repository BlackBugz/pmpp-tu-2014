/*
 * convolution_gpu_tmem.h
 *
 *  Created on: 05/nov/2014
 *      Author: marco
 */

#ifndef CONVOLUTION_GPU_TMEM_H_
#define CONVOLUTION_GPU_TMEM_H_

#include "ppm.h"

void ApplyFilterGPUTMem(PPMImage &destImg, PPMImage &srcImg, const float * kernel, unsigned int kernelSize);

#endif /* CONVOLUTION_GPU_TMEM_H_ */
