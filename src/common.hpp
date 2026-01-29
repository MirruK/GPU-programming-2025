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

struct MaskSpec {
  MaskType type = MaskType::NONE;
  std::string path;
};

// Kernel runner (defined in setup.cu)
void run_kernel(std::string infile, std::string outfile, ShaderType shader_type, MaskSpec mask_spec);