#include<iostream>
#include<tuple>
#include<cmath>
using namespace std;

std::tuple<char, char, char,char> ReadImage(char* Image,std::pair<float,float> coord,int width,int height){
    char R,G,B,A;
    auto [x,y]=coord;
    int x_int=std::round(x);
    int y_int=std::round(y);

    if(x_int<0 || x_int>=width || y_int<0 || y_int>=height){
        R=0;
        G=0;
        B=0;
        A=0;
    }
    else{
        int index=(y_int*width+x_int)*4;
        R=Image[index];
        G=Image[index+1];
        B=Image[index+2];
        A=Image[index+3];
    }

    return std::make_tuple(R,G,B,A);
}