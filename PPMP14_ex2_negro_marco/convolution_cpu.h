#ifndef CONVOLUTION_CPU_H
#define CONVOLUTION_CPU_H

void convolveHCPU(unsigned int *dst, const unsigned int *src,
	const float *kernel, int kernelSize, int w, int h);

void convolveVCPU(unsigned int *dst, const unsigned int *src,
	const float *kernel, int kernelSize, int w, int h);

#endif // CONVOLUTION_CPU_H

