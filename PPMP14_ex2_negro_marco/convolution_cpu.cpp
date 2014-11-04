#include <algorithm>
#include <cmath>

#include "convolution_cpu.h"

void
convolveHCPU(unsigned int *dst, const unsigned int *src,
	const float *kernel, int kernelSize, int w, int h)
{
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			float finalRed = 0.0f;
			float finalGreen = 0.0f;
			float finalBlue = 0.0f;

			for (int i = 0; i < kernelSize; i++)
			{
				int px = x + (i - kernelSize/2);

				// Clamp to [0, w-1]
				px = std::min(px, w-1);
				px = std::max(px, 0);

				unsigned int pixel = src[y * w + px];

				unsigned char r = pixel & 0x000000ff;
				unsigned char g = (pixel & 0x0000ff00) >> 8;
				unsigned char b = (pixel & 0x00ff0000) >> 16;

				finalRed   += r * kernel[i];
				finalGreen += g * kernel[i];
				finalBlue  += b * kernel[i];
			}

			unsigned char finalRed_uc = roundf(finalRed);
			unsigned char finalGreen_uc = roundf(finalGreen);
			unsigned char finalBlue_uc = roundf(finalBlue);

			unsigned int finalPixel = finalRed_uc
				| (finalGreen_uc << 8)
				| (finalBlue_uc << 16);
			dst[y * w + x] = finalPixel;
		}
	}
}

void
convolveVCPU(unsigned int *dst, const unsigned int *src,
	const float *kernel, int kernelSize, int w, int h)
{
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			float finalRed = 0.0f;
			float finalGreen = 0.0f;
			float finalBlue = 0.0f;

			for (int i = 0; i < kernelSize; i++)
			{
				int py = y + (i - kernelSize/2);

				// Clamp to [0, h-1]
				py = std::min(py, h-1);
				py = std::max(py, 0);

				unsigned int pixel = src[py * w + x];

				unsigned char r = pixel & 0x000000ff;
				unsigned char g = (pixel & 0x0000ff00) >> 8;
				unsigned char b = (pixel & 0x00ff0000) >> 16;

				finalRed   += r * kernel[i];
				finalGreen += g * kernel[i];
				finalBlue  += b * kernel[i];
			}

			unsigned char finalRed_uc = roundf(finalRed);
			unsigned char finalGreen_uc = roundf(finalGreen);
			unsigned char finalBlue_uc = roundf(finalBlue);

			unsigned int finalPixel = finalRed_uc
				| (finalGreen_uc << 8)
				| (finalBlue_uc << 16);
			dst[y * w + x] = finalPixel;
		}
	}
}

