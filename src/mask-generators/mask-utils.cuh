#pragma once
#include <cuda_runtime.h>

enum GradientDirection : int {
    LEFT_TO_RIGHT = 0,
    RIGHT_TO_LEFT = 1,
    TOP_TO_BOTTOM = 2,
    BOTTOM_TO_TOP = 3
};

void generate_gradient_mask_GPU(float* mask_d, int w, int h, GradientDirection dir);