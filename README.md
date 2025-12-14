# Project for GPU programming 2025
## Various shader effects implemented in CUDA

## Project description

### Architecture overview

When the main function is ran, the following steps are taken:

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

Use the Makefile provided by running: `make`

This builds the binary, you can then run it using:
`./shader -s SHADER_NAME -f FILENAME... -d DIRNAME`

The outputs will be placed either in the output directory specified by the -d flag,
or into the current directory with the filenames of the outupts in the format output-NNN.ppm