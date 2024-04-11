#include <GetDeviceInfo.hpp>
#include <string>

std::string GetDeviceName(cl_device_id id) {
  size_t size = 0;
  clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);
  std::string result;
  result.resize(size);
  clGetDeviceInfo(id, CL_DEVICE_NAME, size, const_cast<char *>(result.data()),
                  nullptr);
  return result;
}

cl_uint GetDeviceMaxComputeUnits(cl_device_id id) {
  cl_uint result;
  clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &result,
                  nullptr);
  return result;
}

cl_uint GetDeviceMaxClockFrequency(cl_device_id id) {
  cl_uint result;
  clGetDeviceInfo(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &result,
                  nullptr);
  return result;
}

cl_ulong GetDeviceGlobalMemSize(cl_device_id id) {
  cl_ulong result;
  clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &result,
                  nullptr);
  return result;
}

cl_ulong GetDeviceLocalMemSize(cl_device_id id) {
  cl_ulong result;
  clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &result,
                  nullptr);
  return result;
}

std::string GetDeviceVendor(cl_device_id id) {
  size_t size = 0;
  clGetDeviceInfo(id, CL_DEVICE_VENDOR, 0, nullptr, &size);
  std::string result;
  result.resize(size);
  clGetDeviceInfo(id, CL_DEVICE_VENDOR, size, const_cast<char *>(result.data()),
                  nullptr);
  return result;
}

std::string GetDeviceVersion(cl_device_id id) {
  size_t size = 0;
  clGetDeviceInfo(id, CL_DEVICE_VERSION, 0, nullptr, &size);
  std::string result;
  result.resize(size);
  clGetDeviceInfo(id, CL_DEVICE_VERSION, size,
                  const_cast<char *>(result.data()), nullptr);
  return result;
}

cl_uint GetDeviceCacheLineSize(cl_device_id id) {
  cl_uint result;
  clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint),
                  &result, nullptr);
  return result;
}

bool GetDiviceEnableDouble(cl_device_id id) {
  cl_bool result;
  clGetDeviceInfo(id, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_bool), &result,
                  nullptr);
  return result;
}

cl_uint GetDeviceSIMDWidth(cl_device_id id) {
  cl_uint result;
  clGetDeviceInfo(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, sizeof(cl_uint),
                  &result, nullptr);
  return result;
}

size_t GetDeviceMaxWorkGroupSize(cl_device_id id) {
  size_t result;
  clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &result,
                  nullptr);
  return result;
}