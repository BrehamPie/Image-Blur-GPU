#include "cpu_blur.h"
void box_blur_cpu(const unsigned char* input, unsigned char* output, int width, int height, int channels, int radius){
    int kernel_size = 2 * radius + 1;
    int k_half = kernel_size / 2;

    for(int y = 0; y <height; y++){
        for(int x = 0; x < width; x++){
            int sum[3] = {0, 0, 0};
            int count = 0;
            for(int ky = -k_half; ky <=k_half; ky++){
                for(int kx = -k_half; kx <=k_half; kx++){
                    int nx = x + kx;
                    int ny = y + ky;
                    if(nx>=0 and ny>=0 and nx < width and ny <height){
                        int idx = (ny*width + nx) * channels;
                        for(int c = 0; c<channels; c++){
                            sum[c] += input[idx + c];
                        }
                        count++;
                    }
                }
            }
            int out_idx = (y * width + x) * channels;
            for(int c = 0; c<channels; c++){
                output[out_idx + c] = static_cast<unsigned char>(sum[c] / count);
            }
        }
    }
}