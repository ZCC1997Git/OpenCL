#include<iostream>
using namespace std;
char* LoadImage(std::string filename,int& width, int& height);

int main(){
    int width, height;
    auto buffer = LoadImage("./image/LenaRGB.png", width, height);
    
    delete[] buffer;
    return 0;
}