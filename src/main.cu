#include <iostream>
#include <chrono>
#include <filesystem>
#include <vector>
#include <cuda_runtime.h>

#include "utils.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "cpu_blur.h"
#include "gpu_blur.h"

namespace fs = std::filesystem;

int main() {
    const char* image_folder = "image";

    // Iterate over all files in the folder
    for (const auto& entry : fs::directory_iterator(image_folder)) {
        if (!entry.is_regular_file()) continue;

        std::string input_path = entry.path().string();
        std::cout << "Processing image: " << input_path << std::endl;

        int width, height, channels;
        unsigned char* image = load_image(input_path.c_str(), width, height, channels);
        if (!image) {
            std::cerr << "Failed to load image: " << input_path << std::endl;
            continue;
        }

        size_t img_size = static_cast<size_t>(width) * height * channels;
        unsigned char* result = new unsigned char[img_size];

        // Run CPU blur 5 times, accumulate times
        //warm up run
        box_blur_cpu(image, result, width, height, channels, 3);
        double cpu_total_time = 0;
        for (int i = 0; i < 5; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            box_blur_cpu(image, result, width, height, channels, 3);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dur = end - start;
            cpu_total_time += dur.count();
        }
        double cpu_avg_time = cpu_total_time / 5.0;

        // Setup CUDA events once
        cudaEvent_t gpu_start, gpu_stop;
        cudaEventCreate(&gpu_start);
        cudaEventCreate(&gpu_stop);

        // Run GPU blur 5 times, accumulate times
        float gpu_total_time = 0;
        // Warm up run
        box_blur_gpu(image, result, width, height, channels, 3);
        for (int i = 0; i < 5; ++i) {
            cudaEventRecord(gpu_start);
            box_blur_gpu(image, result, width, height, channels, 3);
            cudaEventRecord(gpu_stop);
            cudaEventSynchronize(gpu_stop);
            float ms;
            cudaEventElapsedTime(&ms, gpu_start, gpu_stop);
            gpu_total_time += ms;
        }
        float gpu_avg_time = gpu_total_time / 5.0f;

        // Clean up CUDA events
        cudaEventDestroy(gpu_start);
        cudaEventDestroy(gpu_stop);
        // Save one result image (optional, can skip if you want)
        std::string output_path = entry.path().parent_path().string() + "/output_" + entry.path().filename().string();
        // if (save_image(output_path.c_str(), result, width, height, channels)) {
        //     std::cout << "Saved output image to " << output_path << std::endl;
        // }

        // Print timings
        std::cout << "Average CPU Blur Time: " << cpu_avg_time << " ms" << std::endl;
        std::cout << "Average GPU Blur Time: " << gpu_avg_time << " ms" << std::endl;

        // Cleanup
        stbi_image_free(image);
        delete[] result;
        std::cout << "----------------------------------------" << std::endl;
    }

    return 0;
}
