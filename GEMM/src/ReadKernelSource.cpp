#include <fstream>
#include <iostream>
#include <opencl.hpp>
#include <opencl_step.hpp>
#include <string>
#include <typeindex>

std::string ReadKernelSource(std::string filename, std::type_index type) {
    std::ifstream fin(filename, std::ios::in);
    std::string Kernel((std::istreambuf_iterator<char>(fin)),
                       std::istreambuf_iterator<char>());
    fin.close();

    if (type == typeid(float)) {
        Kernel = "#define T float\n" + Kernel;
    } else if (type == typeid(double)) {
        Kernel = "#define T double\n" + Kernel;
        Kernel = "#pragma OPENCL EXTENSION cl_khr_fp64:enable\n" + Kernel;
    } else {
        std::cerr << "Invalid type" << std::endl;
        exit(-1);
    }
    return Kernel;
}