#include <opencl.hpp>
#include <iostream>
#include <string>
#include <vector>

cl_int BuildProgram(cl_program program, cl_uint num_device, std::vector<cl_device_id> device, std::string options = "")
{
    cl_int errNum = clBuildProgram(program, 1, device.data(), options.data(), nullptr, nullptr);
    if (errNum != CL_SUCCESS)
    {
        char buildLog[16384];
        clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, nullptr);
        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return errNum;
    }
    return errNum;
}