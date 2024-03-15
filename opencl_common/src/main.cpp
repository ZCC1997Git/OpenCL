#include <opencl.hpp>
#include <iostream>
#include <opencl_step.hpp>
#include <GetPlatformInfo.hpp>
#include <GetDeviceInfo.hpp>

int main()
{
    std::vector<cl_device_id> device;
    cl_platform_id platform = GetPlatform(device);
    std::cout << "Platform: " << platform << "\n";
    std::cout << "Platform Name: " << GetPlatformName(platform) << std::endl;
    std::cout << "Platform Vendor: " << GetPlatformVendor(platform) << std::endl;
    std::cout << "Platform Version: " << GetPlatformVersion(platform) << std::endl;
    std::cout << "Platform Profile: " << GetPlatformProfile(platform) << std::endl;
    std::cout << "Platform Extensions: " << GetPlatformExtensions(platform) << std::endl;

    std::cout << std::endl;
    std::cout << "Device: " << device[0] << std::endl;
    std::cout << "Device Name: " << GetDeviceName(device[0]) << std::endl;
    std::cout << "Device Max Computer unit:" << GetDeviceMaxComputeUnits(device[0]) << std::endl;
    std::cout << "Device Max Clock Frequency:" << GetDeviceMaxClockFrequency(device[0]) << std::endl;
    std::cout << "Device Global Mem Size:" << GetDeviceGlobalMemSize(device[0]) << std::endl;
    std::cout << "Device Vendor:" << GetDeviceVendor(device[0]) << std::endl;
    std::cout << "Device Version:" << GetDeviceVersion(device[0]) << std::endl;
    std::cout << "Device Cache Line Size:" << GetDeviceCacheLineSize(device[0]) << std::endl;

    auto context = CreateContext(platform, device, 1);

    return 0;
}