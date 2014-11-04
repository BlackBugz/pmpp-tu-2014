#include <cassert>
#include <cmath>

#include "common.h"

// Initializes the given float array with a univariate Gaussian
void
initKernelGaussian1D(float *kernel, unsigned int kernelSize)
{
	assert(kernel != NULL);
	assert(kernelSize > 0 && (kernelSize & 1) == 1);

	unsigned int mean = kernelSize/2;
	float sigma = ((kernelSize-1) * 0.5f - 1.0f) * 0.3f + 0.8f;

	float sum = 0.0f;
	for (unsigned int i = 0; i < kernelSize; i++)
	{
		int x = i - mean;
		float temp = std::exp(-(x*x / (2.0*sigma*sigma))) / (std::sqrt(2.0*M_PI)*sigma);
		kernel[i] = temp;
		sum += temp;
	}

	// Normalize kernel values
	float rcpSum = 1.0f / sum;
	for (unsigned int i = 0; i < kernelSize; i++)
	{
		kernel[i] *= rcpSum;
	}
}

