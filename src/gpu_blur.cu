#include "gpu_blur.h"
#include <cuda_runtime.h>


__global__
void blur_kernel(const unsigned char* input, unsigned char* output, int width, int height, int channels, int kernel_size){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x>=width or y>=height) return;
    int sum[3] = {};
    int count = 0;
    int radius = kernel_size / 2;
    for(int ky = -radius; ky<=radius; ky++){
        for(int kx = -radius; kx<=radius; kx++){
            int nx = x + kx;
            int ny = y + ky;
            if(nx>=0 and ny>=0 and nx<width and ny < height){
                int idx = (ny*width+nx)*channels;
                for(int c=0;c<channels;c++){
                    sum[c]+=input[idx+c];
                }
                count++;
            }
        }
    }
    int out_idx = (y*width + x) * channels;
    for(int c = 0; c<channels; c++){
        output[out_idx + c] = static_cast<unsigned char>(sum[c]/count);
    }
}
void box_blur_gpu(const unsigned char* input, unsigned char* output, int width, int height, int channels, int kernel_size){
    size_t num_bytes = width * height * channels * sizeof (unsigned char);

    unsigned char* d_input, *d_output;
    cudaMalloc((void**)&d_input, num_bytes);
    cudaMalloc((void**)&d_output, num_bytes);

    cudaMemcpy(d_input, input, num_bytes, cudaMemcpyHostToDevice);
    dim3 threads(32, 32);
    dim3 blocks((width + 31) / 32, (height + 31) / 32);

    blur_kernel<<<blocks, threads>>>(d_input, d_output, width, height, channels, kernel_size);

    cudaMemcpy(output, d_output, num_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}