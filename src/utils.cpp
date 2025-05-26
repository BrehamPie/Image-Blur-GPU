#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "utils.h"
#include <iostream>

unsigned char* load_image(const char* filename, int& width, int& height, int& channels){
    unsigned char* img = stbi_load(filename, &width, &height, &channels, 0);
    if(!img){
        std::cerr << "Failed to load image:"<<filename << " Error: " << stbi_failure_reason() << std::endl;
        exit(EXIT_FAILURE);
    }
    return img;
}
bool save_image(const char* filename, unsigned char* data, int width, int height, int channels){
    int success = stbi_write_png(filename, width, height, channels, data, width * channels);
    if(!success){
        std::cerr << "Failed to save image: " << filename << " Error: " << stbi_failure_reason() << std::endl;
        return false;
    }
    return true;
}