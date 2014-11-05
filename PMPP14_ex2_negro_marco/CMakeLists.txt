project(gaussFilter)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)


cuda_add_executable(gaussFilter
	gaussFilter.cpp
	convolution_cpu.cpp
	#matrixMul_gpu.cu
	convolution_gpu_gmem.cu
	common.cpp
	ppm.cpp)