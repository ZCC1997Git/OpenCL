#include <opencl.hpp>
#include <string>
#include <iostream>

cl_kernel CreateKernel(cl_program program, std::string kernel_name)
{
    cl_int errNum;
    cl_kernel kernel = clCreateKernel(program, kernel_name.data(), &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to create kernel: " << kernel_name << std::endl;
        return nullptr;
    }
    return kernel;
}