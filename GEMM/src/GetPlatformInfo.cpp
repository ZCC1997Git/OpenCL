#include <opencl.hpp>
#include <iostream>
#include <string>
#include <GetPlatformInfo.hpp>

std::string GetPlatformName(cl_platform_id id)
{
    size_t size = 0;
    clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);
    std::string result;
    result.resize(size);
    clGetPlatformInfo(id, CL_PLATFORM_NAME, size, const_cast<char *>(result.data()), nullptr);
    return result;
}

std::string GetPlatformVendor(cl_platform_id id)
{
    size_t size = 0;
    clGetPlatformInfo(id, CL_PLATFORM_VENDOR, 0, nullptr, &size);
    std::string result;
    result.resize(size);
    clGetPlatformInfo(id, CL_PLATFORM_VENDOR, size, const_cast<char *>(result.data()), nullptr);
    return result;
}

std::string GetPlatformVersion(cl_platform_id id)
{
    size_t size = 0;
    clGetPlatformInfo(id, CL_PLATFORM_VERSION, 0, nullptr, &size);
    std::string result;
    result.resize(size);
    clGetPlatformInfo(id, CL_PLATFORM_VERSION, size, const_cast<char *>(result.data()), nullptr);
    return result;
}

std::string GetPlatformProfile(cl_platform_id id)
{
    size_t size = 0;
    clGetPlatformInfo(id, CL_PLATFORM_PROFILE, 0, nullptr, &size);
    std::string result;
    result.resize(size);
    clGetPlatformInfo(id, CL_PLATFORM_PROFILE, size, const_cast<char *>(result.data()), nullptr);
    return result;
}

std::string GetPlatformExtensions(cl_platform_id id)
{
    size_t size = 0;
    clGetPlatformInfo(id, CL_PLATFORM_EXTENSIONS, 0, nullptr, &size);
    std::string result;
    result.resize(size);
    clGetPlatformInfo(id, CL_PLATFORM_EXTENSIONS, size, const_cast<char *>(result.data()), nullptr);
    return result;
}