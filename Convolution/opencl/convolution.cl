
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

template <typename T, int filterSize>
__kernel void convolutionConstant(int imageOutSizeX, int imageOutSizeY,
                                  __global T *imageIn,
                                  __constant T filter[filterSize * filterSize],
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

template <typename T, int filterSize, int BS>
__kernel void convolutionConstantShared(
    int imageOutSizeX, int imageOutSizeY, __global T *imageIn,
    __constant T filter[filterSize * filterSize], __global T *imageOut) {

  __local T l_pixel[BS + filterSize - 1][BS + filterSize - 1];

  int x = get_global_id(0);
  int y = get_global_id(1);
  int tidx = get_local_id(0);
  int tidy = get_local_id(1);
  int imageInSizeX = imageOutSizeX + filterSize - 1;
  /*center*/
  l_pixel[tidy][tidx] = imageIn[y * imageInSizeX + x];
  /*right*/
  if (tidx < filterSize - 1) {
    l_pixel[tidy][tidx + BS] = imageIn[y * imageInSizeX + x + BS];
  }
  if (tidy < filterSize - 1) {
    l_pixel[tidy + BS][tidx] = imageIn[(y + BS) * imageInSizeX + x];
  }

  if (tidx < filterSize - 1 && tidy < filterSize - 1) {
    l_pixel[tidy + BS][tidx + BS] = imageIn[(y + BS) * imageInSizeX + x + BS];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  T sum = 0;

  for (int fy = 0; fy < filterSize; fy++) {
    for (int fx = 0; fx < filterSize; fx++) {
      sum += l_pixel[tidy + fy][tidx + fx] * filter[fy * filterSize + fx];
    }
  }
  imageOut[y * imageOutSizeX + x] = sum;
}

template <typename T, int filterSize, int BS, int BX, int BY>
__kernel void convolutionConstantSharedUnroll(
    int imageOutSizeX, int imageOutSizeY, __global T *imageIn,
    __constant T filter[filterSize * filterSize], __global T *imageOut) {

  __local T l_pixel[BS * BY + filterSize - 1][BS * BX + filterSize - 1];

  __local T l_pixel_ref[BS + filterSize - 1][BS + filterSize - 1];

  int x = get_global_id(0);
  int y = get_global_id(1);
  int tidx = get_local_id(0);
  int tidy = get_local_id(1);
  int block_x = get_group_id(0);
  int block_y = get_group_id(1);
  int imageInSizeX = imageOutSizeX + filterSize - 1;
  /*center*/
  for (int i = 0; i < BX; i++) {
    for (int j = 0; j < BY; j++) {
      l_pixel[tidy * BY + j][tidx * BX + i] =
          imageIn[(block_y * BY * BS + tidy * BY + j) * imageInSizeX +
                  block_x * BS * BY + tidx * BX + i];
    }
  }

  /*right*/
  if (tidx < filterSize - 1) {
    for (int j = 0; j < BY; j++) {
      l_pixel[tidy * BY + j][BS * BX + tidx] =
          imageIn[(block_y * BY * BS + tidy * BY + j) * imageInSizeX +
                  (block_x + 1) * BS * BY + tidx];
    }
  }

  if (tidy < filterSize - 1) {
    for (int i = 0; i < BX; i++) {
      l_pixel[BS * BY + tidy][tidx * BX + i] =
          imageIn[((block_y + 1) * BY * BS + tidy) * imageInSizeX +
                  block_x * BS * BY + tidx * BX + i];
    }
  }

  if (tidx < filterSize - 1 && tidy < filterSize - 1) {
    l_pixel[BS * BY + tidy][BS * BX + tidx] =
        imageIn[((block_y + 1) * BY * BS + tidy) * imageInSizeX +
                (block_x + 1) * BS * BY + tidx];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  T sum[BX * BY] = {0};

  for (int fy = 0; fy < filterSize; fy++) {
    for (int fx = 0; fx < filterSize; fx++) {
      T filterItem = filter[fy * filterSize + fx];
      for (int i = 0; i < BX; i++) {
        for (int j = 0; j < BY; j++) {
          sum[j * BY + i] +=
              l_pixel[tidy * BY + j + fy][tidx * BX + i + fx] * filterItem;
        }
      }
    }
  }

  for (int i = 0; i < BX; i++) {
    for (int j = 0; j < BY; j++) {
      imageOut[(block_y * BY * BS + tidy * BY + j) * imageOutSizeX +
               block_x * BS * BY + tidx * BX + i] = sum[j * BX + i];
    }
  }
}