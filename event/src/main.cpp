#include <GetDeviceInfo.hpp>
#include <GetPlatformInfo.hpp>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <opencl.hpp>
#include <opencl_step.hpp>

int main() {
  std::vector<cl_device_id> device;
  cl_platform_id platform = GetPlatform(device, 1, CL_DEVICE_TYPE_GPU);
  std::cout << "Platform: " << GetPlatformName(platform) << std::endl;
  std::cout << "Device: " << GetDeviceName(device[0]) << std::endl;

  auto context = CreateContext(platform, device, 1);
  auto CommandQueue = CreateCommandQueue(context, device[0]);
  auto KernelSource = ReadKernelSource("./opencl/opencl_event.cl");
  auto Program = CreateProgram(context, device[0], KernelSource);
  BuildProgram(Program, 1, device, "-cl-std=CL2.0");

  /*chose the kernel used*/
  std::string kernel_name[2];
  kernel_name[0] = "kernel1_test";
  kernel_name[1] = "kernel2_test";
  auto Kernel1 = CreateKernel(Program, kernel_name[0]);
  auto Kernel2 = CreateKernel(Program, kernel_name[1]);

  /*set the size of buf*/
  constexpr size_t contentLength = sizeof(int) * 16 * 1024 * 1024;
  cl_mem src1MemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, contentLength,
                                     nullptr, nullptr);
  cl_mem src2MemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, contentLength,
                                     nullptr, nullptr);
  int *pHostBuffer = new int[contentLength / sizeof(int)];
  for (size_t i = 0; i < contentLength / sizeof(int); i++)
    pHostBuffer[i] = i;

  cl_event event1, event2;
  clEnqueueWriteBuffer(CommandQueue, src1MemObj, CL_FALSE, 0, contentLength,
                       pHostBuffer, 0, nullptr, &event1);

  clEnqueueWriteBuffer(CommandQueue, src2MemObj, CL_FALSE, 0, contentLength,
                       pHostBuffer, 1, &event1, &event2);

  cl_mem dstMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, contentLength,
                                    nullptr, nullptr);

  SetKernelArg(Kernel1, dstMemObj, src1MemObj, src2MemObj);
  auto maxWorkGroupSize = GetDeviceMaxWorkGroupSize(device[0]);

  /*wait for the data of src1MemObj and src2MemObj*/
  clWaitForEvents(1, &event2);
  clReleaseEvent(event1);
  clReleaseEvent(event2);
  event1 = nullptr;
  event2 = nullptr;

  /*launch the kernel*/
  size_t globalWorkSize = contentLength / sizeof(int);
  clEnqueueNDRangeKernel(CommandQueue, Kernel1, 1, nullptr, &globalWorkSize,
                         &maxWorkGroupSize, 0, nullptr, &event1);

  /*prepare the sencond kernel*/
  SetKernelArg(Kernel2, dstMemObj, src1MemObj, src2MemObj);
  /*launch the second kernel*/
  clEnqueueNDRangeKernel(CommandQueue, Kernel2, 1, nullptr, &globalWorkSize,
                         &maxWorkGroupSize, 1, &event1, &event2);

  /*check*/
  int *pDeviceBuffer = new int[contentLength / sizeof(int)];
  clEnqueueReadBuffer(CommandQueue, dstMemObj, CL_TRUE, 0, contentLength,
                      pDeviceBuffer, 1, &event2, nullptr);

  for (size_t i = 0; i < contentLength / sizeof(int); i++) {
    int testData = pHostBuffer[i] + pHostBuffer[i];
    testData = testData * pHostBuffer[i] - pHostBuffer[i];
    if (testData != pDeviceBuffer[i]) {
      std::cout << "Error: " << i << std::endl;
      break;
    }
  }
  std::cout << "Test is passed" << std::endl;

  ReleaseSource(context, CommandQueue, Program, Kernel1, Kernel2);
  delete[] pHostBuffer;
  delete[] pDeviceBuffer;
  return 0;
}