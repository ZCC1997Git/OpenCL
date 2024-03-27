#include<string>
#include<cstring>
#include<FreeImage/usr/include/FreeImage.h>
#include<iostream>

char* LoadImage(std::string filename,int& width, int& height){
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filename.c_str(), 0);
    FIBITMAP* image = FreeImage_Load(format, filename.c_str());
    FIBITMAP* temp = image;
    image = FreeImage_ConvertTo32Bits(temp);
    FreeImage_Unload(temp);
    width = FreeImage_GetWidth(image);
    height = FreeImage_GetHeight(image);
    int size = width * height * 4;
    char* buffer= new char[size];
    
    memcpy(buffer, FreeImage_GetBits(image), size);
    FreeImage_Unload(image);
    return buffer;

}