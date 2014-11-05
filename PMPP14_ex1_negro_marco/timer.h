#ifndef TIMER_H
#define TIMER_H

#if !defined(WIN32)

#include <sys/time.h>
inline double
SysGetTime_ms()
{
	struct timeval now;
	gettimeofday(&now, 0);
	double curTime_ms = now.tv_sec * 1000.0 + now.tv_usec / 1000.0;
	return curTime_ms;
}

#else

#include <windows.h>
inline double
SysGetTime_ms()
{
	LARGE_INTEGER tick, ticksPerSecond;
	QueryPerformanceFrequency(&ticksPerSecond);
	QueryPerformanceCounter(&tick);
	double curTime_ms = tick.QuadPart / (double)ticksPerSecond.QuadPart;
	return curTime_ms;
}

#endif

#endif // TIMER_H

