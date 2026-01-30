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
  - Right now uses a FILTER_SIZE\*FILTER_SIZE blur filter

## How to run

Use the Makefile provided by running: `make all`

This builds the binary, you can then run it using:
`./main INPUT_FILENAME.ppm OUTPUT_FILENAME.ppm shader (mask)`

The output is written to the specified OUTPUT_FILENAME.ppm
