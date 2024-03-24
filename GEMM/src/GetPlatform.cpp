#include <iostream>
#include <opencl.hpp>
#include <vector>

cl_platform_id GetPlatform(std::vector<cl_device_id>& device,
                           cl_uint PlatformId = 1,
                           cl_device_type deviceType = CL_DEVICE_TYPE_GPU) {
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id Platform;
    cl_context context = nullptr;

    errNum = clGetPlatformIDs(0, nullptr, &numPlatforms);

    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        std::cerr << "Failed to find any OpenCL platforms" << std::endl;
        return nullptr;
    }

    errNum = clGetPlatformIDs(PlatformId, &Platform, nullptr);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Failed to initial the first platform" << std::endl;
        return nullptr;
    }

    cl_uint numDevices;
    errNum = clGetDeviceIDs(Platform, deviceType, 0, 0, &numDevices);
    if (errNum != CL_SUCCESS || numDevices <= 0) {
        std::cerr << "Failed to find any devices" << std::endl;
        return nullptr;
    }
    device.resize(numDevices);
    errNum = clGetDeviceIDs(Platform, deviceType, numDevices, device.data(),
                            nullptr);

    return Platform;
}
