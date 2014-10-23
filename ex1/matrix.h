#ifndef MATRIX_H
#define MATRIX_H

#include <cstdlib>

typedef struct {
	int width;
	int height;
	size_t pitch; // row size in bytes
	float* elements;
} Matrix;

//----------------------------------------------------------------------------
Matrix AllocateMatrixCPU(int width, int height, bool random);
void FreeMatrixCPU(Matrix &M);

Matrix AllocateMatrixGPU(int width, int height);
void FreeMatrixGPU(Matrix &M);

#endif // MATRIX_H
