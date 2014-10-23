#ifndef MATRIXMUL_CPU_H
#define MATRIXMUL_CPU_H

#include "matrix.h"

/** \brief Matrix multiplication on the CPU
 *  \param M matrix M
 *  \param N matrix N
 *  \param P matrix P
 *
 *  Computes P = M * N
 */
void MatrixMulCPU(const Matrix &M, const Matrix &N, Matrix &P);

#endif // MATRIXMUL_CPU_H
