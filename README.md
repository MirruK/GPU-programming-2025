# Project for GPU programming 2025
## Various shader effects implemented in CUDA

## Project description

### Architecture overview

- img.hpp/img.cpp
  - Image parsing and writing
  - Image related structs and classes
- img-utils.cuh
  - Cuda header file with needed declarations for the kernel and setup code
- setup.cu
  - Setup code for the GPU (mallocs and memcpys)
  - Reading input file (this should be done somewhere else ideally)
  - Writing output file (this should be done somewhere else ideally)
- convolution-kernel.cu
  - The most unoptimized convolution kernel implementation
  - Right now uses a FILTER_SIZE*FILTER_SIZE blur filter

### TODO: Review this architecture and implement

1. Parse cmd-args
2. create or read input image(s)
3. Prepare input / output variables
4. Allocate memory
5. Memcpy input contents to GPU
6. Run the correct shader kernel on the GPU
7. Collect the result in host memory
8. Repeat from step 2. for all images
9. Write output image(s) to disk
10. cleanup

## How to run

Use the Makefile provided by running: `make all`

This builds the binary, you can then run it using:
`./main INPUT_FILENAME.ppm OUTPUT_FILENAME.ppm shader (mask)`

The output is written to the specified OUTPUT_FILENAME.ppm


The outputs will be placed either in the output directory specified by the -d flag,
or into the current directory with the filenames of the outupts in the format output-NNN.ppm
