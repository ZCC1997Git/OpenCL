#include <opencl.hpp>
#include <iostream>
#include <opencl_step.hpp>
#include <GetPlatformInfo.hpp>
#include <GetDeviceInfo.hpp>
#include <cmath>
#include <chrono>

int main()
{
    std::vector<cl_device_id> device;
    cl_platform_id platform = GetPlatform(device);
    std::cout << "Platform: " << GetPlatformName(platform) << std::endl;
    std::cout << "Device" << GetDeviceName(device[0]) << std::endl;

    auto context = CreateContext(platform, device, 1);
    auto CommandQueue = CreateCommandQueue(context, device[0]);
    auto KernelSource = ReadKernelSource("./opencl/gemm.cl");
    auto Program = CreateProgram(context, device[0], KernelSource);
    BuildProgram(Program, 1, device);
    auto Kernel = CreateKernel(Program, "ClGemm");

    constexpr int width = 1024;
    constexpr int height = 1024;
    constexpr int size = width * height;

    cl_float *A = new cl_float[size];
    cl_float *B = new cl_float[size];
    cl_float *C = new cl_float[size];
    cl_float *C_ref = new cl_float[size];

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            A[y * width + x] = y;
            B[y * width + x] = x;
            C[y * width + x] = 0;
        }
    }

    /*create buffer*/
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * size, A, nullptr);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * size, B, nullptr);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * size, nullptr, nullptr);

    /*set kernel arguments*/
    clSetKernelArg(Kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(Kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(Kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(Kernel, 3, sizeof(cl_int), &height);
    clSetKernelArg(Kernel, 4, sizeof(cl_int), &width);

    size_t global[2] = {height, width};
    size_t local[2] = {32, 32};

    auto start1 = std::chrono::system_clock::now();
    auto err = clEnqueueNDRangeKernel(CommandQueue, Kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error in clEnqueueNDRangeKernel: " << err << std::endl;
        return 1;
    }
    clFinish(CommandQueue);
    auto end1 = std::chrono::system_clock::now();
    auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
    std::cout << "GPU Elapsed time: " << elapsed1 << "ms" << std::endl;

    /*read buffer*/
    err = clEnqueueReadBuffer(CommandQueue, bufferC, CL_TRUE, 0, sizeof(cl_float) * size, C, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error in clEnqueueReadBuffer: " << err << std::endl;
        return 1;
    }

    /*reference*/

    auto start2 = std::chrono::system_clock::now();
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float sum = 0;
            for (int k = 0; k < width; k++)
            {
                sum += A[y * width + k] * B[k * width + x];
            }
            C_ref[y * width + x] = sum;
        }
    }
    auto end2 = std::chrono::system_clock::now();
    auto elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
    std::cout << "CPU Elapsed time: " << elapsed2 << "ms" << std::endl;
    std::cout << "Speed Up is:" << (double)elapsed2 / elapsed1 << std::endl;

    /*check*/
    for (int i = 0; i < size; i++)
    {
        if (std::abs(C[i] - C_ref[i]) > 1.0e-3f)
        {
            std::cerr << "Error: " << i << " " << C[i] << " " << C_ref[i] << std::endl;
            return 1;
        }
    }

    ReleaseSource(context, CommandQueue, Program, Kernel);
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_ref;
    return 0;
}