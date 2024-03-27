#include <fstream>
#include <iostream>
#include <opencl.hpp>
#include <opencl_step.hpp>
#include <string>
#include <typeindex>

std::string ReadKernelSource(std::string filename) {
    std::ifstream fin(filename, std::ios::in);
    std::string Kernel((std::istreambuf_iterator<char>(fin)),
                       std::istreambuf_iterator<char>());
    fin.close();
    return Kernel;
}