#include <opencl.hpp>
#include <iostream>
#include <vector>
#include <assert.h>

cl_context CreateContext(cl_platform_id Platform, std::vector<cl_device_id> device, cl_uint num_device)
{
    assert(num_device <= device.size());
    cl_int err;
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)Platform, 0};
    cl_context context = clCreateContext(properties, 1, device.data(), nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error: Failed to create a compute context!" << std::endl;
        return nullptr;
    }
    return context;
}

cl_context CreateContext(cl_platform_id Platform, cl_device_id device)
{
    cl_int err;
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)Platform, 0};
    cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error: Failed to create a compute context!" << std::endl;
        return nullptr;
    }
    return context;
}