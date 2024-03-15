#define CL_TARGET_OPENCL_VERSION 210
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>
#include <string>

cl_program CreateProgram(cl_context context, cl_device_id device, std::string KernelSource)
{
    cl_int errNum;

    const char *KernelSourceChar = KernelSource.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, static_cast<const char **>(&KernelSourceChar), nullptr, &errNum);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Fail to create program" << std::endl;
        return nullptr;
    }

    errNum = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (errNum != CL_SUCCESS)
    {
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, nullptr);
        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return nullptr;
    }

    return program;
}