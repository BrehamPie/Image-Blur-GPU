# CUDA Image Blur Project

This project implements a simple **image blur filter** (box blur) using both CPU and GPU (CUDA) to demonstrate the performance benefits of GPU acceleration for image processing tasks.

---

## Features

- Load and save images using the lightweight [stb_image](https://github.com/nothings/stb) library.
- Apply a box blur filter on images using:
  - A **CPU implementation** (single-threaded).
  - A **CUDA GPU implementation** for massive parallel speedup.
- Supports color (RGB) images with configurable kernel size.
- Processes multiple images from a folder and reports average CPU and GPU execution times.
- Saves output blurred images with a naming convention for easy verification.

---

## Project Structure

.
├── include/           # Header files
├── src/               # Source files (.cpp, .cu)
├── obj/               # Object files (auto-generated)
├── bin/               # Executable output
├── image/             # Input images
├── README.md          # This file


---

## Requirements

- NVIDIA GPU with CUDA support  
- CUDA Toolkit installed  
- C++17 capable compiler (e.g. `g++` or `clang++`)  
- `nvcc` CUDA compiler  
- Make sure to compile with `-std=c++17` and link CUDA libraries  

---

## Build Instructions

Make sure you have CUDA Toolkit installed and a CUDA-capable GPU.

Then simply run:

    make

This will build the executable at `bin/run`.

To clean build files:

    make clean


---

## Usage

1. Place input images (e.g., `.jpg`, `.png`) inside the `image/` folder.  
2. After building, run the program:

    ./bin/run

The program will:

- Process all images in the `image/` folder.  
- Run CPU and GPU blur 5 times each on every image.  
- Print average CPU and GPU execution times per image.  
- Save blurred output images as `output_<original-filename>` in the same folder.

---

## Performance

Typical speedup observed on a mid-range GPU is **10× to 100×** over CPU for medium-sized images (e.g., 512×512). Speedup depends on image size, CPU optimizations, and GPU kernel efficiency.

---

## Next Steps / Improvements

- Implement different convolution kernels for sharpening, edge detection, etc.  
- Optimize GPU kernel using shared memory and better memory coalescing.  
- Add multi-threading to CPU implementation for fairer comparison.  
- Create a GUI or batch processing tool with command line arguments.

---

## License

This project is released under the MIT License.

---

## Acknowledgments

- [stb_image](https://github.com/nothings/stb) for image loading and saving.  
- NVIDIA CUDA Toolkit for GPU programming support.

---

Feel free to explore, modify, and extend this project!  
Questions? Contact me or open an issue.
