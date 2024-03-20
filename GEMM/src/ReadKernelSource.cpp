#include <opencl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <opencl_step.hpp>

std::string ReadKernelSource(std::string filename)
{
    std::ifstream fin(filename, std::ios::in);
    std::string Kernel((std::istreambuf_iterator<char>(fin)),
                       std::istreambuf_iterator<char>());
    fin.close();
    return Kernel;
}