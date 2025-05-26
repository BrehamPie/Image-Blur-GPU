#ifndef CPU_BLUR_H
#define CPU_BLUR_H
void box_blur_cpu(const unsigned char* input, unsigned char* output, int width, int height, int channels, int radius);
#endif // CPU_BLUR_H