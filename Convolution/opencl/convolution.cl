
template <typename T, int filterSize>
__kernel void convolutionNative(int imageOutSizeX, int imageOutSizeY,
                                __global T *imageIn, __global T *filter,
                                __global T *imageOut) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int imageInSizeX = imageOutSizeX + filterSize - 1;

  if (y < imageOutSizeY && x < imageOutSizeX) {
    T sum = 0;
    for (int fy = 0; fy < filterSize; fy++) {
      for (int fx = 0; fx < filterSize; fx++) {
        sum += imageIn[(y + fy) * imageInSizeX + x + fx] *
               filter[fy * filterSize + fx];
      }
    }
    imageOut[y * imageOutSizeX + x] = sum;
  }
}