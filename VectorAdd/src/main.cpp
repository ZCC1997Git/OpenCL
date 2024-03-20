#define CL_TARGET_OPENCL_VERSION 210
#include <iostream>
#include <string>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

std::string ReadKernelSource(std::string filename);
cl_context CreateContext(cl_device_id &device);
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device);
cl_program CreateProgram(cl_context context, cl_device_id device, std::string KernelSource);
bool CreateMemObjects(cl_context context, cl_mem memObjects[3], float *a, float *b, int ArraySize);
void CleanUp(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem memObjects[3]);

int main()
{
    auto KernelSource = ReadKernelSource("./opencl/VectorAdd.cl");

    cl_device_id device;
    auto context = CreateContext(device);

    cl_command_queue commandQueue = CreateCommandQueue(context, device);

    cl_program program = CreateProgram(context, device, KernelSource);

    cl_kernel kernel = clCreateKernel(program, "vector_add", nullptr);

    constexpr int ArraySize = 1024 * 20;

    float A[ArraySize];
    float B[ArraySize];

    for (int i = 0; i < ArraySize; i++)
    {
        A[i] = static_cast<float>(1);
        B[i] = static_cast<float>(2);
    }

    cl_mem memObjects[3] = {nullptr, nullptr, nullptr};

    if (!CreateMemObjects(context, memObjects, A, B, ArraySize))
    {
        std::cerr << "Failed to create memory objects" << std::endl;
        CleanUp(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    auto errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments(" << errNum << ")" << std::endl;
        CleanUp(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    size_t globalWorkSize[1] = {ArraySize};
    size_t localWorkSize[1] = {256};

    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution(" << errNum << ")" << std::endl;
        CleanUp(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    float result[ArraySize];
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, ArraySize * sizeof(float), result, 0, nullptr, nullptr);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer(" << errNum << ")" << std::endl;
        CleanUp(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    float sum = 0;
    for (int i = 0; i < ArraySize; i++)
    {
        sum += result[i];
    }
    std::cout << "Check:" << sum << "==" << 3 * ArraySize << std::endl;

    std::cout << "Executed program successfully" << std::endl;
    CleanUp(context, commandQueue, program, kernel, memObjects);

    return 0;
}