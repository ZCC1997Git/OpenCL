#include <opencl.hpp>
#include <string>
#include <cstring>
#include <FreeImage/include/FreeImage.h>
#include <iostream>

cl_mem LoadImage(cl_context context, std::string filename, int &width, int &height)
{
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filename.c_str(), 0);
    FIBITMAP *image = FreeImage_Load(format, filename.c_str());
    FIBITMAP *temp = image;
    image = FreeImage_ConvertTo32Bits(temp);
    FreeImage_Unload(temp);
    width = FreeImage_GetWidth(image);
    height = FreeImage_GetHeight(image);
    int size = width * height * 4;
    char *buffer = new char[size];

    memcpy(buffer, FreeImage_GetBits(image), size);
    FreeImage_Unload(image);

    cl_image_format clImageFormat;
    clImageFormat.image_channel_order = CL_RGBA;
    clImageFormat.image_channel_data_type = CL_UNORM_INT8;
    cl_image_desc clImageDesc;
    memset(&clImageDesc, 0, sizeof(cl_image_desc));
    clImageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    clImageDesc.image_width = width;
    clImageDesc.image_height = height;
    cl_int error;
    cl_mem clImage = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &clImageFormat, &clImageDesc, buffer, &error);

    if (error != CL_SUCCESS)
    {
        std::cerr << "Error creating image object" << std::endl;
        return 0;
    }
    delete[] buffer;
    return clImage;
}