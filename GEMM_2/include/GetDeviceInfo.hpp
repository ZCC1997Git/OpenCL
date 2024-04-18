#pragma once
#include <opencl.hpp>
#include <string>

std::string GetDeviceName(cl_device_id id);

cl_uint GetDeviceMaxComputeUnits(cl_device_id id);

cl_uint GetDeviceMaxClockFrequency(cl_device_id id);

cl_ulong GetDeviceGlobalMemSize(cl_device_id id);
cl_ulong GetDeviceLocalMemSize(cl_device_id id);

std::string GetDeviceVendor(cl_device_id id);

std::string GetDeviceVersion(cl_device_id id);

cl_uint GetDeviceCacheLineSize(cl_device_id id);

bool GetDiviceEnableDouble(cl_device_id id);

cl_uint GetDeviceSIMDWidth(cl_device_id id);
size_t GetDeviceMaxWorkGroupSize(cl_device_id id);