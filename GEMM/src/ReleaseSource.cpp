#include <iostream>
#include <opencl.hpp>

void ReleaseSource(cl_context context,
                   cl_command_queue commandQueue,
                   cl_program program,
                   cl_kernel kernel) {
    if (commandQueue != nullptr) {
        clReleaseCommandQueue(commandQueue);
    }

    if (kernel != nullptr) {
        clReleaseKernel(kernel);
    }

    if (program != nullptr) {
        clReleaseProgram(program);
    }

    if (context != nullptr) {
        clReleaseContext(context);
    }
}
