#include <opencl.hpp>
#include <iostream>
#include <opencl_step.hpp>
#include <GetPlatformInfo.hpp>
int main()
{
    std::vector<cl_device_id> device;
    cl_platform_id platform = GetPlatform(device);
    std::cout << "Platform Name: " << GetPlatformName(platform) << std::endl;
    std::cout << "Platform Vendor: " << GetPlatformVendor(platform) << std::endl;
    std::cout << "Platform Version: " << GetPlatformVersion(platform) << std::endl;
    std::cout << "Platform Profile: " << GetPlatformProfile(platform) << std::endl;
    std::cout << "Platform Extensions: " << GetPlatformExtensions(platform) << std::endl;

    auto context = CreateContext(device, 1);

    return 0;
}