#include<FreeImage/usr/include/FreeImage.h>
#include<cstring>
#include<string>
#include<iostream>

void SaveImage(std::string filename, char* buffer, int width, int height) {
    FIBITMAP* image = FreeImage_ConvertFromRawBits((BYTE*)buffer, width, height, width * 4, 32, 0xFF000000, 0x00FF0000, 0x0000FF00);
    FreeImage_Save(FIF_PNG, image, filename.c_str());
    FreeImage_Unload(image);
    
    std::cout<<"Image saved as "<<filename<<std::endl;
}