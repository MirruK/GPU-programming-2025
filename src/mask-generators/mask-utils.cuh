#pragma once
#include <cuda_runtime.h>
#include "../common.hpp" // For the GradientDirection definition

void generate_gradient_mask_GPU(float* mask_d, int w, int h, GradientDirection dir);
void generate_radial_mask_GPU(float* mask_d, int w, int h, int inner_percent, int outer_percent);