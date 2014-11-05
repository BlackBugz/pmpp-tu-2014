#ifndef MATRIXMUL_GPU_H
#define MATRIXMUL_GPU_H

#include "matrix.h"

void MatrixMulGPU(const Matrix &M, const Matrix &N, Matrix &P);
void MatrixMulGPUSimple(const Matrix &M, const Matrix &N, Matrix &P);

#endif // MATRIXMUL_GPU_H
