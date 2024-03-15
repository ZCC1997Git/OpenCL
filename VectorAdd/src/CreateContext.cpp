#define CL_TARGET_OPENCL_VERSION 210
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>

cl_context CreateContext(cl_device_id &device)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = nullptr;

    errNum = clGetPlatformIDs(0, nullptr, &numPlatforms);

    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms" << std::endl;
        return nullptr;
    }
    else
    {

        std::cout << "The number of platform is " << numPlatforms << std::endl;
    }

    errNum = clGetPlatformIDs(1, &firstPlatformId, nullptr);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to initial the first platform" << std::endl;
        return nullptr;
    }

    errNum = clGetDeviceIDs(firstPlatformId, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "There is no GPU, trying CPU..." << std::endl;
        errNum = clGetDeviceIDs(firstPlatformId, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);

        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Fail to find any CPU and GPU in this platform" << std::endl;
            return nullptr;
        }
    }
    else
    {
        std::cout << "Find the GPU" << std::endl;
    }

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Fail to create context" << std::endl;
        return nullptr;
    }
    return context;
}