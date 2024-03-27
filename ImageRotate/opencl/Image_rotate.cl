/* set sampler*/
__constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;

/*
 *@brief opencl kernel for image rotation
 *@param srcImg input image
 *@param dstImg output image
 *@param angle angle of rotation
 */
__kernel void image_rotate(__read_only image2d_t srcImg,
                           __write_only image2d_t dstImg, float angle) {
  /*get the width and height of image */
  int width = get_image_width(srcImg);
  int height = get_image_height(srcImg);
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  float sinmap = sin(angle);
  float cosmap = cos(angle);
  /*calculate the raotation center */
  int hwidth = width / 2;
  int hheight = height / 2;
  int xt = x - hwidth;
  int yt = y - hheight;
  /*calculate the coordinate after rotation */
  float2 readCoord = (float2)(cosmap * xt - sinmap * yt + hwidth,
                              sinmap * xt + cosmap * yt + hheight);
  /*read the pixel value from input image */
  float4 pixel = read_imagef(srcImg, sampler, readCoord);
  /*write the pixel value to output image */
  write_imagef(dstImg, (int2)(x, y), pixel);
}
