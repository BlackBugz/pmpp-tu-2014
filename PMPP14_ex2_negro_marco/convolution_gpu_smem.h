/*
 * convolution_gpu_gmem.h
 *
 *  Created on: 05/nov/2014
 *      Author: marco
 */

#ifndef CONVOLUTION_GPU_GMEM_H_
#define CONVOLUTION_GPU_GMEM_H_

#include "ppm.h"

void ApplyFilterGPUSMem(PPMImage &srcImg, PPMImage &destImg, const float * kernel, unsigned int kernelSize);

#endif /* CONVOLUTION_GPU_GMEM_H_ */
