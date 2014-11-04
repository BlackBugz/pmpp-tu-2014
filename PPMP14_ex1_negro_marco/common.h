#ifndef COMMON_H
#define COMMON_H

#include <cstdio>

#include <cuda_runtime.h>

//----------------------------------------------------------------------------
#define CUDA_SUCCEEDED(call)										\
{																	\
	cudaError_t err = call;											\
																	\
	if (cudaSuccess != err) {										\
		std::printf("CUDA error in %s:%i: %s (%i)\n",				\
				__FILE__, __LINE__, cudaGetErrorString(err), err);	\
		std::exit(EXIT_FAILURE);									\
	}																\
}

//----------------------------------------------------------------------------
inline unsigned int
divUp(unsigned int dividend, unsigned int divisor)
{
	unsigned int result = dividend / divisor;
	if (dividend % divisor) ++result;
	return result;
}

#endif // COMMON_H
