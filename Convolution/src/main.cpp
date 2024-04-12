#include <GetDeviceInfo.hpp>
#include <GetPlatformInfo.hpp>
#include <chrono>
#include <cmath>
#include <convolution.hpp>
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
  auto KernelSource = ReadKernelSource("./opencl/convolution.cl");
  auto KernelNames = ParseKernelFromSource(KernelSource);
  InstanceTemplate(KernelNames[0], KernelSource, "int", "5");
  InstanceTemplate(KernelNames[1], KernelSource, "int", "5");
  InstanceTemplate(KernelNames[2], KernelSource, "int", "5","16");
  InstanceTemplate(KernelNames[3], KernelSource, "int", "5","16","2","2");
   std::cout << "The kernel name is "<< std::endl;
  for(auto &name:KernelNames){
    std::cout<<name<<std::endl;
  }
  auto Program = CreateProgram(context, device[0], KernelSource);
  BuildProgram(Program, 1, device, "-cl-std=CL2.0");

  auto Kernel1 = CreateKernel(Program, KernelNames[3]);

  /*the convolution related*/
  constexpr int width = 1024 * 4 + 4;
  constexpr int heigt = 1024 * 4 + 4;
  constexpr size_t filterSize = 5;
  constexpr int imageOutSizeX = width - filterSize + 1;
  constexpr int imageOutSizeY = heigt - filterSize + 1;

  int Flilter[filterSize * filterSize] = {-1, 1,  1,  1,  -1, -1, 1,  1, 1,
                                          -1, -1, 1,  -2, 1,  1,  -1, 1, 1,
                                          1,  -1, -1, 1,  1,  1,  1};

  int *LayerOut = new int[imageOutSizeX * imageOutSizeY];
  int *LayerOut_ref = new int[imageOutSizeX * imageOutSizeY];
  char *ImageOut = new char[imageOutSizeX * imageOutSizeY * 4];
  int *TheLayer = new int[width * heigt];

  cl_mem FlilterBuffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * filterSize * filterSize, Flilter, nullptr);

  cl_mem TheLayerBuffer = clCreateBuffer(
      context, CL_MEM_READ_ONLY, sizeof(int) * width * heigt, nullptr, nullptr);

  cl_mem LayerOutBuffer = clCreateBuffer(
      context, CL_MEM_WRITE_ONLY, sizeof(int) * imageOutSizeX * imageOutSizeY,
      nullptr, nullptr);

  SetKernelArg(Kernel1, imageOutSizeX, imageOutSizeX, TheLayerBuffer,
               FlilterBuffer, LayerOutBuffer);
  for (int layer = 0; layer < 3; layer++) {
    for (int i = 0; i < width * heigt; i++) {
      TheLayer[i] = layer + 1 + i % 255;
    }
    clEnqueueWriteBuffer(CommandQueue, TheLayerBuffer, CL_TRUE, 0,
                         sizeof(int) * width * heigt, TheLayer, 0, nullptr,
                         nullptr);
    size_t globalWorkSize[2] = {imageOutSizeX/2, imageOutSizeY/2};
    size_t localWorkSize[2] = {16, 16};
    cl_event event;
    clEnqueueNDRangeKernel(CommandQueue, Kernel1, 2, nullptr, globalWorkSize,
                           localWorkSize, 0, nullptr, &event);
    /*get the during time*/

    clWaitForEvents(1, &event);
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                            &start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                            &end, nullptr);
    auto elapsed1 = (end - start) / 1e6;
    std::cout << "The opencl elapsed time is " << elapsed1 << "ms" << std::endl;

    clEnqueueReadBuffer(CommandQueue, LayerOutBuffer, CL_TRUE, 0,
                        sizeof(int) * imageOutSizeX * imageOutSizeY, LayerOut,
                        0, nullptr, nullptr);

    /*check*/
    auto time_start = std::chrono::high_resolution_clock::now();
    Convolution_unroll_simd_avx<filterSize,2,2>(width, heigt, TheLayer, Flilter, LayerOut_ref);
    auto time_end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                      time_end - time_start)
                      .count();
    std::cout << "The cpu elapsed time is " << elapsed << "ms" << std::endl;
    for (int i = 0; i < imageOutSizeX * imageOutSizeY; i++) {
      if (LayerOut[i] != LayerOut_ref[i]) {
        std::cout << "Error in layer " << layer << " at " << i << " "
                  << LayerOut[i] << " " << LayerOut_ref[i] << std::endl;
        return 1;
      }
    }
    std::cout << "Layer " << layer << " is correct" << std::endl;
    for (int i = 0; i < imageOutSizeX * imageOutSizeY; i++) {
      ImageOut[i * 4 + layer] = LayerOut[i];
    }
  }

  ReleaseSource(context, CommandQueue, Program, Kernel1);
  delete[] TheLayer;
  delete[] LayerOut;
  delete[] LayerOut_ref;
  delete[] ImageOut;

  return 0;
}