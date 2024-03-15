#define CL_TARGET_OPENCL_VERSION 210
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>

void CleanUp(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != nullptr)
        {
            clReleaseMemObject(memObjects[i]);
        }
    }

    if (commandQueue != nullptr)
    {
        clReleaseCommandQueue(commandQueue);
    }

    if (kernel != nullptr)
    {
        clReleaseKernel(kernel);
    }

    if (program != nullptr)
    {
        clReleaseProgram(program);
    }

    if (context != nullptr)
    {
        clReleaseContext(context);
    }
}