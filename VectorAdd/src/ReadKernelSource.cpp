#define CL_TARGET_OPENCL_VERSION 210
#include <iostream>
#include <fstream>
#include <string>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

std::string ReadKernelSource(std::string filename)
{
    std::ifstream fin(filename, std::ios::in);
    std::string Kernel((std::istreambuf_iterator<char>(fin)),
                       std::istreambuf_iterator<char>());
    fin.close();
    return Kernel;
}