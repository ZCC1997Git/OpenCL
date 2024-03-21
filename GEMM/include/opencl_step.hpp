#pragma once
#include <opencl.hpp>
#include <string>
#include <tuple>
#include <typeindex>
#include <vector>

std::string ReadKernelSource(std::string filename, std::type_index type);

cl_platform_id GetPlatform(std::vector<cl_device_id>& device,
                           cl_uint PlatformId = 1,
                           cl_device_type deviceType = CL_DEVICE_TYPE_GPU);

cl_context CreateContext(cl_platform_id Platform,
                         std::vector<cl_device_id> device,
                         cl_uint num_device);
cl_context CreateContext(cl_platform_id Platform, cl_device_id device);

std::vector<cl_command_queue> CreateCommandQueue(
    cl_context context,
    std::vector<cl_device_id>& device);
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device);

cl_program CreateProgram(cl_context context,
                         cl_device_id device,
                         std::string KernelSource);
cl_int BuildProgram(cl_program program,
                    cl_uint num_device,
                    std::vector<cl_device_id> device,
                    std::string options = "");

cl_kernel CreateKernel(cl_program program, std::string kernel_name);

void ReleaseSource(cl_context context,
                   cl_command_queue commandQueue,
                   cl_program program,
                   cl_kernel kernel);

template <class Lam, class Args, size_t... Is>
constexpr void Detail_Loop(Lam lam,
                           cl_kernel kernel,
                           Args&& ArgTuple,
                           std::index_sequence<Is...>) {
    (lam(kernel, Is, std::get<Is>(ArgTuple)), ...);
};

template <class Args>
void SetSpecificKernelArg(cl_kernel Kernel, size_t dim, Args&& arg) {
    using T = typename std::remove_reference_t<decltype(arg)>;
    clSetKernelArg(Kernel, dim, sizeof(T), &arg);
}

template <class... Args>
void SetKernelArg(cl_kernel Kernel, Args&&... args) {
    auto ArgTuple = std::make_tuple(args...);
    constexpr size_t num_args = sizeof...(Args);

    auto lambda = [](auto kernel, auto dim, auto&& para) {
        SetSpecificKernelArg(kernel, dim, std::forward<decltype(para)>(para));
    };

    Detail_Loop(lambda, Kernel, std::forward<decltype(ArgTuple)>(ArgTuple),
                std::make_index_sequence<num_args>{});
}

template <class... BUF>
void ReleaseBuffer(BUF&&... buf) {
    (clReleaseMemObject(buf), ...);
}

template <class... Args>
void ReleaseCPUBuf(Args&&... args) {
    (delete[] args, ...);
}