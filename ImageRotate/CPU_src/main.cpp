#include <iostream>
#include <tuple>
#include <cmath>
#include <chrono>
using namespace std;
char *LoadImage(std::string filename, int &width, int &height);
std::tuple<char, char, char, char> ReadImage(char *Image, std::pair<float, float> coord, int width, int height);
void SaveImage(std::string filename, char *buffer, int width, int height);

int main()
{
    int width, height;
    auto buffer = LoadImage("./image/LenaRGB.png", width, height);
    char *after = new char[width * height * 4];
    float angle = 45.0 * 3.1415926 / 180.0;
    float sinmap = sin(angle);
    float cosmap = cos(angle);
    /*calculate the raotation center */
    int hwidth = width / 2;
    int hheight = height / 2;
    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            int xt = i - hwidth;
            int yt = j - hheight;

            auto x = cosmap * xt - sinmap * yt + hwidth;
            auto y = sinmap * xt + cosmap * yt + hheight;

            auto [R, G, B, A] = ReadImage(buffer, {x, y}, width, height);
            after[(j * width + i) * 4] = R;
            after[(j * width + i) * 4 + 1] = G;
            after[(j * width + i) * 4 + 2] = B;
            after[(j * width + i) * 4 + 3] = A;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    /*output the time in unit ms*/
    std::cout << "Execution time: " << diff.count() * 1000 << "ms" << std::endl;

    SaveImage("./image/LenaRGB_rotation_cpu.png", after, width, height);
    delete[] buffer;
    delete[] after;
    return 0;
}