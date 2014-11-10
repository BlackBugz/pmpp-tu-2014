#include <cstdlib>
#include <cstdio>

#include "common.h"
#include "convolution_cpu.h"
#include "convolution_gpu_gmem.h"
#include "ppm.h"
#include "timer.h"

int
main(int argc, char **argv)
{
	if(argc < 3){
		printf("Wrong number of arguments\n");
		return EXIT_FAILURE;
	}

	char * inFileName = argv[1];
	int kernelSize = atoi(argv[2]);

	if((kernelSize & 1) == 0){
		printf("WARNING: kernel size must be odd\n");
		return EXIT_FAILURE;
	}

	printf("Launching test with following inputs: \n");
	printf("Filename:\t%s\n", inFileName);
	printf("Kernel Size:\t%i\n", kernelSize);

	// TODO Task 1: Load image
	PPMImage *original = new PPMImage();
	original->loadBin(inFileName);

	// TODO Task 1: Generate gaussian filter kernel
	float * filter = new float[kernelSize];
	initKernelGaussian1D(filter, kernelSize);

	float * filterGPU = AllocateGaussianGPU(kernelSize);
	CopyGaussianToDevice(filter, filterGPU, kernelSize);

	// TODO Task 1: Blur image on CPU

	// initializing data for CPU computation
	PPMImage *srcImg = new PPMImage(*original);
	PPMImage *destImg = new PPMImage(srcImg->height, srcImg->width);

	// performing CPU computation
	convolveHCPU(destImg->data, srcImg->data, filter, kernelSize, srcImg->width, srcImg->height);
	*srcImg = *destImg;
	convolveVCPU(destImg->data, srcImg->data, filter, kernelSize, srcImg->width, srcImg->height);

	// saving CPU task results and freeing memory
	printf("Done. Saving result in out_cpu.ppm\n");
	destImg->saveBin("out_cpu.ppm");
	delete destImg;
	delete srcImg;


	// TODO Task 2: Blur image on GPU (Global memory)
	
	// initializing data for GPU (global memory) computation
	PPMImage srcGPUG = AllocateImageGPU(original->width, original->height);
	PPMImage destGPUG = AllocateImageGPU(original->width, original->height);
	CopyToDeviceImage(*original, srcGPUG);

	// perform computation on GPU
	ApplyFilterGPUGMem(srcGPUG, destGPUG, filterGPU, kernelSize);

	PPMImage *destImgG = new PPMImage(original->width, original->height);
	CopyToHostImage(destGPUG, *destImgG);
	destImgG->saveBin("out_gpu_gmem.ppm");
	FreeImageGPU(srcGPUG);
	FreeImageGPU(destGPUG);

	// TODO Task 3: Blur image on GPU (Shared memory)

	// TODO Task 4: Blur image on GPU (Constant memory)
	
	// TODO Task 5: Blur image on GPU (L1/texture cache)
	
	// TODO Task 6: Blur image on GPU (all memory types)

	FreeGaussianGPU(filterGPU);
	return EXIT_SUCCESS;
}

