#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>

#include "ppm.h"
#include "common.h"

bool
PPMImage::loadBin(const char *fn)
{
	if (!fn)
		return false;

	std::ifstream imgFile(fn, std::ifstream::binary);

	if (imgFile.fail())
	{
		std::printf("ERROR: Couldn't open \"%s\"\n", fn);
		return false;
	}

	char line[256];
	imgFile.getline(line, sizeof(line));

	if (line[0] != 'P' || line[1] != '6')
	{
		std::printf("ERROR: Invalid identification string \"%s\"\n", line);
		return false;
	}

	// Skip comment
	if (imgFile.peek() == '#')
		imgFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	imgFile >> width;
	imgFile >> height;

	unsigned short maxValue;
	imgFile >> maxValue;
	if (maxValue > 255)
	{
		std::printf("ERROR: maxValue > 255 is unsupported\n");
		return false;
	}

	// Skip \n after maxValue
	imgFile.ignore(1, '\n');

	// Read pixels into temporary buffer
	unsigned char *temp = new unsigned char[width * height * 3];
	imgFile.read(reinterpret_cast<char*>(temp), width * height * 3);

	// Convert RGB to RGBA
	data = new unsigned int[width * height];
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int srcIdx = y * width * 3 + x * 3;
			unsigned char r = temp[srcIdx + 0];
			unsigned char g = temp[srcIdx + 1];
			unsigned char b = temp[srcIdx + 2];
			unsigned int rgba = r | (g << 8) | (b << 16);
			data[y * width + x] = rgba;
		}
	}

	delete[] temp;

	std::printf("Loaded PPM image \"%s\" (%ix%i)\n", fn, width, height);

	return true;
}

//----------------------------------------------------------------------------
bool
PPMImage::saveBin(const char *fn) const
{
	if (!fn || (width == 0) || (height == 0) || !data)
		return false;

	std::ofstream imgFile(fn, std::ofstream::binary);

	if (imgFile.fail())
	{
		std::printf("ERROR: Couldn't open \"%s\"\n", fn);
		return false;
	}

	imgFile << "P6" << std::endl;
	imgFile << width << std::endl;
	imgFile << height << std::endl;
	imgFile << "255" << std::endl;

	// Convert RGBA to RGB
	unsigned char *temp = new unsigned char[width * height * 3];
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int dstIdx = y * width * 3 + x * 3;
			unsigned int rgba = data[y * width + x];
			unsigned char r = rgba & 0x000000ff;
			unsigned char g = (rgba & 0x0000ff00) >> 8;
			unsigned char b = (rgba & 0x00ff0000) >> 16;
			temp[dstIdx + 0] = r;
			temp[dstIdx + 1] = g;
			temp[dstIdx + 2] = b;
		}
	}

	imgFile.write(reinterpret_cast<const char*>(temp), width * height * 3);
	delete[] temp;

	return true;
}

/*
 * Methods to handle images on the GPU and to copy data.
 */

PPMImage AllocateImageGPU(int width, int height)
{
	PPMImage Idevice;
	Idevice.width = width;
	Idevice.height = height;

	int size = width*height*sizeof(unsigned int);
	CUDA_SUCCEEDED(cudaMalloc(&Idevice.data, size));

	return Idevice;
}

void FreeImageGPU(PPMImage &I)
{
	CUDA_SUCCEEDED(cudaFree(I.data));
}

void CopyToDeviceImage(PPMImage &Idevice, const PPMImage &Ihost)
{
	unsigned int size = Ihost.width * Ihost.height * sizeof(unsigned int);
	CUDA_SUCCEEDED(cudaMemcpy(Idevice.data, Ihost.data, size, cudaMemcpyHostToDevice));
}

void CopyToHostImage(PPMImage &Ihost, const PPMImage &Idevice)
{
	int size = Idevice.width * Idevice.height * sizeof(unsigned int);
	CUDA_SUCCEEDED(cudaMemcpy(Ihost.data, Idevice.data, size, cudaMemcpyDeviceToHost));
}
