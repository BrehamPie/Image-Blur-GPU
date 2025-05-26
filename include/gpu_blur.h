#ifndef GPU_BLUR_H
#define GPU_BLUR_H

void box_blur_gpu(const unsigned char *input, unsigned char *output, int width, int height, int channels, int radius);
#endif // GPU_BLUR_H