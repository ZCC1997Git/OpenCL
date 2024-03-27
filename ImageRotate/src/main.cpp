#include <omp.h>
#include <GetDeviceInfo.hpp>
#include <GetPlatformInfo.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencl.hpp>
#include <opencl_step.hpp>
#include <cstring>
#include <FreeImage/include/FreeImage.h>

int main()
{
    std::vector<cl_device_id> device;
    cl_platform_id platform = GetPlatform(device, 1, CL_DEVICE_TYPE_GPU);
    std::cout << "Platform: " << GetPlatformName(platform) << std::endl;
    std::cout << "Device: " << GetDeviceName(device[0]) << std::endl;

    auto context = CreateContext(platform, device, 1);
    auto CommandQueue = CreateCommandQueue(context, device[0]);
    auto KernelSource =
        ReadKernelSource("./opencl/Image_rotate.cl");
    auto Program = CreateProgram(context, device[0], KernelSource);
    BuildProgram(Program, 1, device, "-cl-std=CL2.0");

    /*chose the kernel used*/
    std::string kernel_name = "image_rotate";
    auto Kernel = CreateKernel(Program, kernel_name);

    /*imageobject[0]:original image; imageobject:after rotation*/
    int width, height;
    cl_mem imageObject[2];
    imageObject[0] = LoadImage(context, "./image/LenaRGB.png", width, height);

    /*create the output image object*/
    cl_image_format clImageFormat;
    clImageFormat.image_channel_order = CL_RGBA;
    clImageFormat.image_channel_data_type = CL_UNORM_INT8;
    cl_image_desc clImageDesc;
    memset(&clImageDesc, 0, sizeof(cl_image_desc));
    clImageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    clImageDesc.image_width = width;
    clImageDesc.image_height = height;
    cl_int error;
    imageObject[1] = clCreateImage(context, CL_MEM_WRITE_ONLY, &clImageFormat, &clImageDesc, nullptr, &error);
    if (error != CL_SUCCESS)
    {
        std::cerr << "Error creating image object" << std::endl;
        return 0;
    }
    float angle = 45.0 * 3.1415926 / 180.0;
    SetKernelArg(Kernel, imageObject[0], imageObject[1], angle);
    size_t global_work_size[2] = {(size_t)width, (size_t)height};
    cl_event event;
    clEnqueueNDRangeKernel(CommandQueue, Kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, &event);
    clFinish(CommandQueue);
    /*calculate the excutation time of kernel*/
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
    std::cout << "Execution time: " << (end - start) / 1000000.0 << "ms" << std::endl;

    /*copy the image after rotation*/
    char *buffer = new char[width * height * 4];
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {(size_t)width, (size_t)height, 1};
    clEnqueueReadImage(CommandQueue, imageObject[1], CL_TRUE, origin, region, 0, 0, buffer, 0, nullptr, nullptr);
    SaveImage("./image/LenaRGB_rotate.png", buffer, width, height);

    ReleaseSource(context, CommandQueue, Program, Kernel);
    delete[] buffer;
    return 0;
}