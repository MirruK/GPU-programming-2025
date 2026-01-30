#pragma once
#include <string>

enum class ShaderType {
  BLUR_BOX,
  BLUR_GAUSSIAN,
  BLUR_MOTION,
  BLUR_SOBEL,
  DITHER,
  GRAYSCALE,
  INVERSION,
  MIRROR
};

enum class MaskType {
  NONE,
  GRADIENT,
  RADIAL,
  PPM
};

enum class GradientDirection : int {
    LEFT_TO_RIGHT,
    RIGHT_TO_LEFT,
    TOP_TO_BOTTOM,
    BOTTOM_TO_TOP
};

struct ShaderSpec {
    ShaderType type = ShaderType::BLUR_BOX;

    // dither params
    int dither_levels = 4;
};

struct MaskSpec {
    MaskType type = MaskType::NONE;

    // gradient params
    GradientDirection gradient_dir = GradientDirection::LEFT_TO_RIGHT;

    // radial params
    int radial_inner = 60;
    int radial_outer = 100;

    // .ppm params
    std::string path;
};

// Kernel runner (defined in setup.cu)
void run_kernel(std::string infile, std::string outfile, ShaderSpec shader_spec, MaskSpec mask_spec);