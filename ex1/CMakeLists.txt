project(matrixMul)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

cuda_add_executable(matrixMul
	matrix.cpp
	matrixMul_cpu.cpp
	matrixMul_gpu.cu
	matrixMul.cpp)