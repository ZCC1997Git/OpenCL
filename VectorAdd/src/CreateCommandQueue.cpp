#define CL_TARGET_OPENCL_VERSION 210
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device)
{
    cl_int errNum;
    cl_command_queue commandQueue = nullptr;

    cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES, 0, 0};
    commandQueue = clCreateCommandQueueWithProperties(context, device, properties, &errNum);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to create command queue" << std::endl;
        return nullptr;
    }

    return commandQueue;
}