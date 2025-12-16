#pragma once
#include "img.hpp"
#include <cstdint>

// Adds two numbers, ensuring there is no overflow, and clamps the resulting value to fit within (0, color_depth)
// Unfortunately I can't use std::clamp here because I am not sure it works if ran on device
__host__ __device__
inline uint16_t clamping_add(uint16_t v1, uint16_t v2, uint16_t color_depth) {
    uint32_t sum = uint32_t(v1) + uint32_t(v2);
    return sum > color_depth ? color_depth : static_cast<uint16_t>(sum);
}

__host__ __device__
inline uint16_t clamping_mul(uint16_t v, float s, uint16_t color_depth) {
  uint32_t product = v * s;
  return product > color_depth ? color_depth : static_cast<uint16_t>(product);
}

__host__ __device__
inline PPMPixel px_add(PPMPixel p1, PPMPixel p2, uint16_t color_depth){
  return {
    clamping_add(p1.r, p2.r, color_depth),
    clamping_add(p1.g, p2.g, color_depth),
    clamping_add(p1.b, p2.b, color_depth),
  };
}

__host__ __device__
inline PPMPixel px_scale(PPMPixel p1, float s, uint16_t color_depth){
  return {
    clamping_mul(p1.r, s, color_depth),
    clamping_mul(p1.g, s, color_depth),
    clamping_mul(p1.b, s, color_depth),
  };
}

void blur_image_GPU(PPMPixel* src_img_d, PPMPixel* dst_img_d, int w, int h, int color_depth);
