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
Matrix
AllocateMatrixCPU(int width, int height, bool random)
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
void
FreeMatrixCPU(Matrix &M)
{
	delete[] M.elements;
	M.elements = NULL;
}

//----------------------------------------------------------------------------

/** \brief Initialize and allocate memory for a Matrix struct on the GPU
 *  \param width matrix width
 *  \param height matrix height
 */
Matrix
AllocateMatrixGPU(int width, int height)
{
	// TODO: Task 4
	Matrix Mdevice;
	Mdevice.width = width;
	Mdevice.height = height;

	cudaMallocPitch((void**)&Mdevice.elements,&Mdevice.pitch, Mdevice.width, Mdevice.height);

	return Mdevice;
}

//----------------------------------------------------------------------------
/** \brief Free the memory of a Matrix struct on the GPU
 *  \param M matrix to free
 */
void
FreeMatrixGPU(Matrix &M)
{
	// TODO: Task 4
	cudaFree(M.elements);
}

/** \brief Copy a host matrix to a device matrix
 *	\param Mdevice matrix related to the space allocated on the device
 *	\param Mhost matrx allocated on the host
 */
void 
 CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
 {
 	int size = Mhost.width * Mhost.height * sizeof(float);
 	cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
 }

/** \brief Copy a device matrix to a host matrix
 *	\param Mhost matrx allocated on the host
 *	\param Mdevice matrix related to the space allocated on the device
 */
void 
 CopyToHostMatrix(Matrix Mhost, const Matrix Mdevice)
 {
 	int size = Mdevice.width * Mdevice.height * sizeof(float);
 	cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
 }

 