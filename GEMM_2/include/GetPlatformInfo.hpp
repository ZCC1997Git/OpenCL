#pragma once
#include <opencl.hpp>
#include <string>
std::string GetPlatformName(cl_platform_id id);

std::string GetPlatformVendor(cl_platform_id id);

std::string GetPlatformVersion(cl_platform_id id);

std::string GetPlatformProfile(cl_platform_id id);

std::string GetPlatformExtensions(cl_platform_id id);