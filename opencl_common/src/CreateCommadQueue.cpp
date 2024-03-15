#include <opencl.hpp>
#include <iostream>
#include <vector>

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

std::vector<cl_command_queue> CreateCommandQueue(cl_context context, std::vector<cl_device_id> &device)
{
    cl_int errNum;
    std::vector<cl_command_queue> commandQueue;
    for (auto &dev : device)
    {
        cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES, 0, 0};
        commandQueue.push_back(clCreateCommandQueueWithProperties(context, dev, properties, &errNum));
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create command queue" << std::endl;
            return std::vector<cl_command_queue>();
        }
    }
    return commandQueue;
}