#ifndef UTILS_H
#define UTILS_H
unsigned char* load_image(const char* filename, int& width, int& height, int& channels);
bool save_image(const char* filename, unsigned char* data, int width, int height, int channels);
#endif // UTILS_H