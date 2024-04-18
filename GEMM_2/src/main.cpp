#include <omp.h>
#include <Gemm.hpp>
#include <GetDeviceInfo.hpp>
#include <GetPlatformInfo.hpp>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <opencl.hpp>
#include <opencl_step.hpp>

int main() {
    std::vector<cl_device_id> device;
    cl_platform_id platform = GetPlatform(device, 1, CL_DEVICE_TYPE_GPU);
    std::cout << "Platform: " << GetPlatformName(platform) << std::endl;
    std::cout << "Device: " << GetDeviceName(device[0]) << std::endl;

    auto context = CreateContext(platform, device, 1);
    auto CommandQueue = CreateCommandQueue(context, device[0]);
    auto KernelSource = ReadKernelSource("./opencl/gemm.cl");
    auto KernelNames = ParseKernelFromSource(KernelSource);
    InstanceTemplate(KernelNames[0], KernelSource, "float");
    InstanceTemplate(KernelNames[1], KernelSource, "16");
    InstanceTemplate(KernelNames[2], KernelSource, "16");
    InstanceTemplate(KernelNames[3], KernelSource, "16");
    InstanceTemplate(KernelNames[4], KernelSource, "16");

    std::cout << "The kernel name is " << std::endl;
    for (auto& name : KernelNames) {
        std::cout << name << std::endl;
    }
    auto Program = CreateProgram(context, device[0], KernelSource);
    BuildProgram(Program, 1, device, "-cl-std=CL2.0");

    auto Kernel1 = CreateKernel(Program, KernelNames[4]);

    constexpr int M = 1024*2;
    constexpr int K = 1024*2;
    constexpr int N = 1024*2;

    float* a = static_cast<float*>(aligned_alloc(32, M * K * sizeof(float)));
    float* b = static_cast<float*>(aligned_alloc(32, K * N * sizeof(float)));
    float* c = static_cast<float*>(aligned_alloc(32, M * N * sizeof(float)));
    float* c_ref =
        static_cast<float*>(aligned_alloc(32, M * N * sizeof(float)));

    for (int i = 0; i < M * K; i++) {
        a[i] = 1.0f / K;
        b[i] = 1.0f;
    }

    cl_mem Device_A =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * M * K, a, nullptr);
    cl_mem Device_B =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * K * N, b, nullptr);

    cl_mem Device_C = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     sizeof(float) * M * N, nullptr, nullptr);

    SetKernelArg(Kernel1, Device_A, Device_B, Device_C, M, N, K);

    size_t globalWorkSize[2] = {N/4, M/4};
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
    auto elapsed1 = (end - start) / 1.0e6;
    std::cout << "The opencl elapsed time is " << elapsed1 << "ms" << std::endl;

    clEnqueueReadBuffer(CommandQueue, Device_C, CL_TRUE, 0,
                        sizeof(float) * M * N, c, 0, nullptr, nullptr);

    /*check*/
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMul_avx<M, K, N>(a, b, c_ref);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto time_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_cpu - start_cpu)
                        .count();

    std::cout << "CPU runing time:" << time_cpu << "ms" << std::endl;
    for (int i = 0; i < M * N; i++) {
        if (std::abs(c[i] - c_ref[i]) > 1e-6) {
            std::cerr << "Error: c[" << i << "] = " << c[i] << std::endl;
            return 1;
        }
    }
    std::cout << "Success!" << std::endl;
    ReleaseSource(context, CommandQueue, Program, Kernel1);
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}