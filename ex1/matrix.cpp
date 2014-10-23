#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "common.h"
#include "matrix.h"

//----------------------------------------------------------------------------

/** \brief Initialize and allocate memory for a Matrix struct on the CPU
 *  \param width matrix width
 *  \param height matrix height
 *  \param random if set, fill matrix with random values using rand(), otherwise
 *                leave the values uninitialized
 */
Matrix AllocateMatrixCPU(int width, int height, bool random)
{
	Matrix M;
	M.width = width;
	M.height = height;
	M.pitch = width * sizeof(float);
	M.elements = new float[width * height];

	if (!random)
		return M;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			M.elements[y * M.pitch / sizeof(float) + x] = (float)rand() / RAND_MAX;
		}
	}

	return M;
}

//----------------------------------------------------------------------------

/** \brief Free the memory of a Matrix struct on the CPU
 *  \param M matrix to free
 */
void FreeMatrixCPU(Matrix &M)
{
	delete[] M.elements;
	M.elements = NULL;
}

//----------------------------------------------------------------------------

/** \brief Initialize and allocate memory for a Matrix struct on the GPU
 *  \param width matrix width
 *  \param height matrix height
 */
Matrix AllocateMatrixGPU(int width, int height)
{
	// TODO: Task 4
	Matrix Mdevice;
	Mdevice.width = width;
	Mdevice.height = height;

	CUDA_SUCCEEDED(cudaMallocPitch((void**)&Mdevice.elements,&Mdevice.pitch, Mdevice.width, Mdevice.height));

	return Mdevice;
}

//----------------------------------------------------------------------------
/** \brief Free the memory of a Matrix struct on the GPU
 *  \param M matrix to free
 */
void FreeMatrixGPU(Matrix &M)
{
	// TODO: Task 4
	CUDA_SUCCEEDED(cudaFree(M.elements));
}

/** \brief Copy a host matrix to a device matrix
 *	\param Mdevice matrix related to the space allocated on the device
 *	\param Mhost matrx allocated on the host
 */
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
	int size = Mhost.width * Mhost.height * sizeof(float);
	CUDA_SUCCEEDED(cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice));
}

/** \brief Copy a device matrix to a host matrix
 *	\param Mhost matrx allocated on the host
 *	\param Mdevice matrix related to the space allocated on the device
 */
void CopyToHostMatrix(Matrix Mhost, const Matrix Mdevice)
{
	int size = Mdevice.width * Mdevice.height * sizeof(float);
	CUDA_SUCCEEDED(cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost));
}

//----------------------------------------------------------------------------

/** \brief Prints the matrix given as input
 * 	\param m matrix to print out
 */
void printMatrix(const Matrix m)
{
	for(int i = 0; i < m.height; ++i){
		for(int j = 0; j < m.width; j++){
			printf("%f ", m.elements[i*m.width+j]);
		}
		printf("\n");
	}
}

/**	\brief Checks if the matrix is empty
 * 	\param m matrix to check
 */
void checkMatrix(const Matrix m)
{
	int count = 0;
	long int c =0;
	int min = 44444444;
	for(int i = 0; i < m.height; ++i){
		for(int j = 0; j < m.width; j++){
			c +=m.elements[i*m.width+j];
			if(m.elements[i*m.width+j] != 0){
				count++;
			}
			if(m.elements[i*m.width+j] < min){
				min = m.elements[i*m.width+j];
			}
		}
	}
	printf("Sum: %lu - != 0: %i  - min: %i\n", c, count, min);
}

/**	\brief Compare the two given matrix
 * 	\param M1 first matrix to compare
 * 	\param M2 second matrix to compare
 */
void compareMatrix(Matrix M1, Matrix M2)
{
	if(M1.width == M2.width && M1.height == M2.height){
		double diff = fabs(M1.elements[0]-M2.elements[0]);
		double maxDiff = diff, minDiff = diff, avg = 0;

		for(int i = 0; i < M1.height; ++i){
			for (j = 0; j < M1.width; ++j) {
				diff = fabs(M1.elements[i * M1.pitch + j] - M2.elements[i * M2.pitch + j]);
				maxDiff = (diff > maxDiff) ? diff : MaxDiff;
				minDiff = (diff < minDiff) ? diff : minDiff;
				avg += diff;
			}
		}

		avg = avg / (M1.width * M1.height);

		printf("Pitch > CPU: %i | GPU %i", M1.pitch, M2.pitch);

		printf("Max difference: %f\n", maxDiff);
		printf("Min difference: %f\n", minDiff);
		printf("Average difference: %f\n", avg);
	} else {
		printf("Matrixes have different sizes, not comparable:\n");
		printf("M1[%i,%i]\n", M1.width, M1.height);
		printf("M2[%i,%i]\n", M2.width, M2.height);
	}
}
 
/**	\brief Initialize a matrix as an identity
 * 	\param M matrix to initialize
 */
void initIdentityMatrix(Matrix M)
{
	for(int i = 0; i < M.height; ++i){
		for (j = 0; j < M.width; ++j) {
			M.elements[i * M.pitch + j] = (i == j) ? 1.0 : 0;
		}
	}
}

/**	\brief Initialize a matrix as an identity. Presumes that the matrix
 * 		has already been initialized with 0s
 *	\param M matrix to initialize
 */
void initIdentityMatrixQuick(Matrix M)
{
	int dim = (M.width < M.height) ? M.width : M.height;
	for(int i = 0; i < dim; ++i){
		M.elements[i * M.pitch + i] = 1.0;
	}
}
