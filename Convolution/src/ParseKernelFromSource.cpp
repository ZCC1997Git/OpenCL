#include <iostream>
#include <string>
#include <vector>

std::vector<std::string> ParseKernelFromSource(std::string KernelSource) {
  std::vector<std::string> kernel_name;
  size_t pos = 0;
  while (pos < KernelSource.size()) {
    size_t start = KernelSource.find("__kernel", pos);
    if (start == std::string::npos) {
      break;
    }
    /*find the space*/
    auto kernel_end = KernelSource.find("(", pos);

    auto kernel_begin = KernelSource.rfind(" ", kernel_end);
    kernel_name.push_back(
        KernelSource.substr(kernel_begin + 1, kernel_end - kernel_begin - 1));
    pos = kernel_end;
  }

  for (auto &name : kernel_name) {
    std::cout << name << std::endl;
  }
  return kernel_name;
}