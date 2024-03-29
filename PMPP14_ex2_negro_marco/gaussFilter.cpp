#include <cstdlib>
#include <cstdio>

#include "common.h"
#include "convolution_cpu.h"
#include "convolution_gpu_gmem.h"
#include "convolution_gpu_cmem.h"
#include "convolution_gpu_smem.h"
#include "convolution_gpu_tmem.h"
#include "convolution_gpu_final.h"
#include "ppm.h"
#include "timer.h"


void printCUDADevices()
{
	// TODO: Task 2

	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	printf("You have %i devices\n", deviceCount);
	for(int i = 0; i < deviceCount; ++i){
		printf("Prop of device %i:\n",i);
		printf("---------------------\n");

		cudaDeviceProp prop;

		cudaGetDeviceProperties(&prop, i);

		printf("Name:\t%s\n", prop.name);
		printf("Compute capability:\t%i.%i\n", prop.major, prop.minor);
		printf("Multiprocessors count:\t%i\n", prop.multiProcessorCount);
		printf("GPU clock rate:\t%i KHz\n", prop.clockRate);
		printf("Total global memory:\t%lu bytes\n", prop.totalGlobalMem);
		printf("L2 cache size:\t%i bytes\n", prop.l2CacheSize);
		printf("\n");
	}
}

int
main(int argc, char **argv)
{
	if(argc < 3){
		printf("Wrong number of arguments\n");
		return EXIT_FAILURE;
	}

	printCUDADevices();

	char * inFileName = argv[1];
	int kernelSize = atoi(argv[2]);

	if((kernelSize & 1) == 0 || (kernelSize < 1) || (kernelSize > 129)){
		printf("WARNING: kernel size must be odd and between 1 and 129\n");
		return EXIT_FAILURE;
	}

	printf("Launching test with following inputs: \n");
	printf("Filename:\t%s\n", inFileName);
	printf("Kernel Size:\t%i\n", kernelSize);

	// TODO Task 1: Load image
	PPMImage original = PPMImage();
	original.loadBin(inFileName);

	// TODO Task 1: Generate gaussian filter kernel
	float filter[kernelSize];
	initKernelGaussian1D(filter, kernelSize);

	float * filterGPU = AllocateGaussianGPU(kernelSize);
	CopyGaussianToDevice(filter, filterGPU, kernelSize);

	// TODO Task 1: Blur image on CPU
	/*
	printf("Starting CPU\n");
	PPMImage srcImg = PPMImage(original);
	PPMImage destImg = PPMImage(srcImg.width, srcImg.height);

	convolveHCPU(destImg.data, srcImg.data, filter, kernelSize, srcImg.width, srcImg.height);
	srcImg = destImg;
	convolveVCPU(destImg.data, srcImg.data, filter, kernelSize, srcImg.width, srcImg.height);

	destImg.saveBin("out_cpu.ppm");
	printf("Done CPU\n");*/


	// TODO Task 2: Blur image on GPU (Global memory)
	printf("Starting GPU on global memory\n");
	PPMImage srcGPUG = AllocateImageGPU(original.width, original.height);
	PPMImage destGPUG = AllocateImageGPU(original.width, original.height);
	CopyToDeviceImage(srcGPUG, original);

	ApplyFilterGPUGMem(destGPUG, srcGPUG, filterGPU, kernelSize);

	PPMImage destImgG = PPMImage(original.width, original.height);
	CopyToHostImage(destImgG, destGPUG);
	destImgG.saveBin("out_gpu_gmem.ppm");

	FreeImageGPU(srcGPUG);
	FreeImageGPU(destGPUG);
	printf("Done GPU on global memory\n");

	// TODO Task 3: Blur image on GPU (Shared memory)
	printf("Starting GPU on shared memory\n");
	PPMImage srcGPUS = AllocateImageGPU(original.width, original.height);
	PPMImage destGPUS = AllocateImageGPU(original.width, original.height);
	CopyToDeviceImage(srcGPUS, original);

	ApplyFilterGPUGMem(destGPUS, srcGPUS, filterGPU, kernelSize);

	PPMImage destImgS = PPMImage(original.width, original.height);
	CopyToHostImage(destImgS, destGPUS);
	destImgS.saveBin("out_gpu_smem.ppm");

	FreeImageGPU(srcGPUS);
	FreeImageGPU(destGPUS);
	printf("Done GPU on shared memory\n");

	// TODO Task 4: Blur image on GPU (Constant memory)
	printf("Starting GPU on constant memory\n");
	PPMImage srcGPUC = AllocateImageGPU(original.width, original.height);
	PPMImage destGPUC = AllocateImageGPU(original.width, original.height);
	CopyToDeviceImage(srcGPUC, original);

	ApplyFilterGPUCMem(destGPUC, srcGPUC, filter, kernelSize);

	PPMImage destImgC = PPMImage(original.width, original.height);
	CopyToHostImage(destImgC, destGPUC);
	destImgG.saveBin("out_gpu_cmem.ppm");

	FreeImageGPU(srcGPUC);
	FreeImageGPU(destGPUC);
	printf("Done GPU with constant memory\n");
	

	// TODO Task 5: Blur image on GPU (L1/texture cache)
	printf("Starting GPU on texture memory\n");
	PPMImage srcGPUT = AllocateImageGPU(original.width, original.height);
	PPMImage destGPUT = AllocateImageGPU(original.width, original.height);
	CopyToDeviceImage(srcGPUT, original);

	ApplyFilterGPUTMem(destGPUT, srcGPUT, filterGPU, kernelSize);

	PPMImage destImgT = PPMImage(original.width, original.height);
	CopyToHostImage(destImgT, destGPUT);
	destImgT.saveBin("out_gpu_tmem.ppm");

	FreeImageGPU(srcGPUT);
	FreeImageGPU(destGPUT);
	printf("Done GPU with texture memory\n");
	
	// TODO Task 6: Blur image on GPU (all memory types)
	printf("Starting GPU final\n");
	PPMImage srcGPUF = AllocateImageGPU(original.width, original.height);
	PPMImage destGPUF = AllocateImageGPU(original.width, original.height);
	CopyToDeviceImage(srcGPUF, original);
	printf("applying\n");
	ApplyFilterGPUFinal(destGPUF, srcGPUF, filter, kernelSize);
	printf("done\n");
	PPMImage destImgF = PPMImage(original.width, original.height);
	CopyToHostImage(destImgF, destGPUF);
	destImgT.saveBin("out_gpu_final.ppm");
	printf("saving\n");
	FreeImageGPU(srcGPUF);
	FreeImageGPU(destGPUF);
	printf("Done GPU final\n");

	FreeGaussianGPU(filterGPU);
	printf("Finished \n");
	return EXIT_SUCCESS;
}

