#ifndef PPM_H
#define PPM_H

#include <cstring>
#include <algorithm>

struct PPMImage
{
	PPMImage()
	 : width(0),
	   height(0),
	   data(NULL)
	{ }

	PPMImage(int w, int h)
	 : width(w),
	   height(h),
	   data(new unsigned int[w*h])
	{ }

	PPMImage(const PPMImage &other)
	{
		width = other.width;
		height = other.height;
		data = new unsigned int[width * height];
		std::memcpy(data, other.data, width * height * sizeof(unsigned int));
	}

	~PPMImage()
	{
		delete[] data;
	}

	PPMImage& operator=(PPMImage other)
	{
		std::swap(width, other.width);
		std::swap(height, other.height);
		std::swap(data, other.data);

		return *this;
	}

	bool loadBin(const char *fn);
	bool saveBin(const char *fn) const;

	int width;
	int height;
	unsigned int *data;
};

PPMImage AllocateImageGPU(int width, int height);
void FreeImageGPU(PPMImage &I);

void CopyToDeviceImage(PPMImage Ihost, PPMImage Idevice);
void CopyToHostImage(PPMImage Idevice, PPMImage Ihost);

#endif // PPM_H
