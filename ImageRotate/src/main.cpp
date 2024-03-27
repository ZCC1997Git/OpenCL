#include <omp.h>
#include <GetDeviceInfo.hpp>
#include <GetPlatformInfo.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencl.hpp>
#include <opencl_step.hpp>

/*define the type*/
using Type = cl_float;
/*if enable to check the result*/
#define CHECK_RESULT 0

int main() {
    std::vector<cl_device_id> device;
    cl_platform_id platform = GetPlatform(device, 1, CL_DEVICE_TYPE_GPU);
    std::cout << "Platform: " << GetPlatformName(platform) << std::endl;
    std::cout << "Device" << GetDeviceName(device[0]) << std::endl;
    std::cout << "Enable double:" << GetDiviceEnableDouble(device[0])
              << std::endl;
    std::cout << "Shared Memory: " << GetDeviceLocalMemSize(device[0])
              << std::endl;

    auto context = CreateContext(platform, device, 1);
    auto CommandQueue = CreateCommandQueue(context, device[0]);
    auto KernelSource =
        ReadKernelSource("./opencl/gemm.cl", std::type_index(typeid(Type)));
    auto Program = CreateProgram(context, device[0], KernelSource);
    BuildProgram(Program, 1, device, "-cl-std=CL2.0");

    /*chose the kernel used*/
    std::string all_kernel_name[3] = {"ClGemm", "ClGemm_block",
                                      "ClGemm_block_newversion"};
    std::string kernel_name = all_kernel_name[0];
    auto Kernel = CreateKernel(Program, kernel_name);

    constexpr int width = 1024 * 1;
    constexpr int height = 1024 * 1;
    constexpr int size = width * height;

    Type* A = new Type[size];
    Type* B = new Type[size];
    Type* C = new Type[size];
    Type* C_ref = new Type[size];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            A[y * width + x] = 1;
            B[y * width + x] = 1;
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

    size_t global[2], local[2];

    if (kernel_name == "ClGemm_block_newversion") {
        if (std::type_index(typeid(Type)) != typeid(float)) {
            std::cerr
                << "Error in type: ClGemm_block_newversion only support float"
                << std::endl;
            return 1;
        }
        global[0] = height / 4;
        global[1] = width / 4;
        local[0] = 32 / 4;
        local[1] = 32 / 4;
    } else if (kernel_name == "ClGemm_block" || kernel_name == "ClGemm") {
        global[0] = height;
        global[1] = width;
        local[0] = 32;
        local[1] = 32;
    } else {
        std::cerr << "Error in kernel name" << std::endl;
        return 1;
    }

    cl_event event;
    if (clEnqueueNDRangeKernel(CommandQueue, Kernel, 2, nullptr, global, local,
                               0, nullptr, &event) != CL_SUCCESS) {
        std::cerr << "Error in clEnqueueNDRangeKernel " << std::endl;
        return 1;
    }
    /*wait the event*/
    clWaitForEvents(1, &event);
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                            &start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                            &end, nullptr);
    auto elapsed1 = (end - start) / 1e6;

    std::cout << "GPU Elapsed time: " << elapsed1 << "ms" << std::endl;
    std::cout << "Bandwith:"
              << 3 * sizeof(Type) * height * width * 1000 / elapsed1 / 1024 /
                     1024 / 1024
              << "GB/s" << std::endl;
    std::cout << "Floats per second:"
              << 2.0 * height * width * height / elapsed1 * 1e3 / 1024 / 1024 /
                     1024
              << "GFLOPS" << std::endl;

    /*read buffer*/
    if (clEnqueueReadBuffer(CommandQueue, bufferC, CL_TRUE, 0,
                            sizeof(Type) * size, C, 0, nullptr,
                            nullptr) != CL_SUCCESS) {
        std::cerr << "Error in clEnqueueReadBuffer" << std::endl;
        return 1;
    }

    /*reference*/
#if (CHECK_RESULT == 1)
    auto start2 = std::chrono::system_clock::now();
#pragma omp parallel for
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
#pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (std::abs(C[i * width + j] - C_ref[i * width + j]) > 1e-3) {
                std::cerr << "Error in " << i << " " << j << std::endl;
                std::cerr << C[i * width + j] << " " << C_ref[i * width + j]
                          << std::endl;
            }
        }
    }
#endif
    ReleaseSource(context, CommandQueue, Program, Kernel);
    ReleaseBuffer(bufferA, bufferB, bufferC);
    ReleaseCPUBuf(A, B, C, C_ref);
    return 0;
}