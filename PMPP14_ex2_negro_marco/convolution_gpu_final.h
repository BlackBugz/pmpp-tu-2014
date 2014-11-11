/*
 * convolution_gpu_final.h
 *
 *  Created on: 05/nov/2014
 *      Author: marco
 */

#ifndef CONVOLUTION_GPU_FINAL_H_
#define CONVOLUTION_GPU_FINAL_H_

#include "ppm.h"

void ApplyFilterGPUFinal(PPMImage &srcImg, PPMImage &destImg, const float * kernel, unsigned int kernelSize);

#endif /* CONVOLUTION_GPU_FINAL_H_ */
