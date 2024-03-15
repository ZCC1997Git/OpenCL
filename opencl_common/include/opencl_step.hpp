#pragma once
#include <opencl.hpp>
#include <string>
#include <vector>

std::string ReadKernelSource(std::string filename);

cl_platform_id GetPlatform(std::vector<cl_device_id> &device, cl_uint PlatformId = 1, cl_device_type deviceType = CL_DEVICE_TYPE_GPU);

cl_context CreateContext(std::vector<cl_device_id> device, cl_uint num_device);
cl_context CreateContext(cl_device_id device);

std::vector<cl_command_queue> CreateCommandQueue(cl_context context, std::vector<cl_device_id> &device);
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device);

cl_program CreateProgram(cl_context context, cl_device_id device, std::string KernelSource);