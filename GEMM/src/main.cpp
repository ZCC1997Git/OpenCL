#include <GetDeviceInfo.hpp>
#include <GetPlatformInfo.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencl.hpp>
#include <opencl_step.hpp>
/*define the type*/
using Type = cl_double;

int main() {
    std::vector<cl_device_id> device;
    cl_platform_id platform = GetPlatform(device);
    std::cout << "Platform: " << GetPlatformName(platform) << std::endl;
    std::cout << "Device" << GetDeviceName(device[0]) << std::endl;

    auto context = CreateContext(platform, device, 1);
    auto CommandQueue = CreateCommandQueue(context, device[0]);
    auto KernelSource =
        ReadKernelSource("./opencl/gemm.cl", std::type_index(typeid(Type)));
    auto Program = CreateProgram(context, device[0], KernelSource);
    BuildProgram(Program, 1, device);
    auto Kernel = CreateKernel(Program, "ClGemm");

    constexpr int width = 1024;
    constexpr int height = 1024;
    constexpr int size = width * height;

    Type* A = new Type[size];
    Type* B = new Type[size];
    Type* C = new Type[size];
    Type* C_ref = new Type[size];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            A[y * width + x] = y;
            B[y * width + x] = x;
            C[y * width + x] = 0;
            C_ref[y * width + x] = 0;
        }
    }

    /*create buffer*/
    cl_mem bufferA =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(Type) * size, A, nullptr);
    cl_mem bufferB =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(Type) * size, B, nullptr);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(Type) * size, nullptr, nullptr);

    /*set kernel arguments*/
    SetKernelArg(Kernel, bufferA, bufferB, bufferC, height, width);

    size_t global[2] = {height, width};
    size_t local[2] = {32, 32};

    auto start1 = std::chrono::system_clock::now();
    if (clEnqueueNDRangeKernel(CommandQueue, Kernel, 2, nullptr, global, local,
                               0, nullptr, nullptr) != CL_SUCCESS) {
        std::cerr << "Error in clEnqueueNDRangeKernel " << std::endl;
        return 1;
    }
    clFinish(CommandQueue);
    auto end1 = std::chrono::system_clock::now();
    auto elapsed1 =
        std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1)
            .count();
    std::cout << "GPU Elapsed time: " << elapsed1 << "ms" << std::endl;

    /*read buffer*/
    if (clEnqueueReadBuffer(CommandQueue, bufferC, CL_TRUE, 0,
                            sizeof(Type) * size, C, 0, nullptr,
                            nullptr) != CL_SUCCESS) {
        std::cerr << "Error in clEnqueueReadBuffer" << std::endl;
        return 1;
    }

    /*reference*/
    auto start2 = std::chrono::system_clock::now();
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Type sum = 0;
            for (int k = 0; k < width; k++) {
                sum += A[y * width + k] * B[k * width + x];
            }
            C_ref[y * width + x] = sum;
        }
    }
    auto end2 = std::chrono::system_clock::now();
    auto elapsed2 =
        std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2)
            .count();
    std::cout << "CPU Elapsed time: " << elapsed2 << "ms" << std::endl;
    std::cout << "Speed Up is:" << (double)elapsed2 / elapsed1 << std::endl;

    /*check*/
    for (int i = 0; i < size; i++) {
        if (std::abs(C[i] - C_ref[i]) > 1.0e-3) {
            std::cerr << "Error: " << i << " " << C[i] << " " << C_ref[i]
                      << std::endl;
            return 1;
        }
    }

    ReleaseSource(context, CommandQueue, Program, Kernel);
    ReleaseBuffer(bufferA, bufferB, bufferC);
    ReleaseCPUBuf(A, B, C, C_ref);
    return 0;
}