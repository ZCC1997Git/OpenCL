#pragma once
#define CL_TARGET_OPENCL_VERSION 210
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif