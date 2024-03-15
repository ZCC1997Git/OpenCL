#define CL_TARGET_OPENCL_VERSION 210
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>

bool CreateMemObjects(cl_context context, cl_mem memObjects[3], float *a, float *b, int SizeArray)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * SizeArray, a, nullptr);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * SizeArray, b, nullptr);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * SizeArray, nullptr, nullptr);

    if (memObjects[0] == nullptr || memObjects[1] == nullptr || memObjects[2] == nullptr)
    {
        std::cerr << "Error creating memory objects" << std::endl;
        return false;
    }

    return true;
}