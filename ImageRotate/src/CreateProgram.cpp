#include <iostream>
#include <opencl.hpp>
#include <string>

cl_program CreateProgram(cl_context context,
                         cl_device_id device,
                         std::string KernelSource) {
    cl_int errNum;

    const char* KernelSourceChar = KernelSource.c_str();
    cl_program program = clCreateProgramWithSource(
        context, 1, static_cast<const char**>(&KernelSourceChar), nullptr,
        &errNum);

    if (errNum != CL_SUCCESS) {
        std::cerr << "Fail to create program" << std::endl;
        return nullptr;
    }

    return program;
}