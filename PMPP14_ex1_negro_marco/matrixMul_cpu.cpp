#include "matrix.h"
#include "matrixMul_cpu.h"

void MatrixMulCPU(const Matrix &M, const Matrix &N, Matrix &P)
{
	// TODO: Task 3
	
	for(int i = 0; i < M.height; ++i){
		for(int j = 0; j < N.width; ++j){
			double sum = 0;
			for(int k = 0; k < M.width;  ++k){
				double m = M.elements[i * M.width + k];
				double n = N.elements[k * N.width + j];
				sum += m*n;
			}
			P.elements[i * M.width + j] = sum;
		}
	}
}

