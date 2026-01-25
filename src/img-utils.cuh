#pragma once
#include "img.hpp"
#include "common.hpp"
#include <cstdint>
#define FILTER_SIZE 3
#define FILTER_RADIUS 1

cudaError_t set_blur_filter(const float h_filter[FILTER_SIZE][FILTER_SIZE]);

void blur_image_GPU(PPMPixel* src_img_d, PPMPixel* dst_img_d, int w, int h, int color_depth);

void select_blur_filter(BlurType type);

void grayscale_image_GPU(PPMPixel* src_img_d, PPMPixel* dst_img_d, float* mask, int w, int h, int color_depth);
void grayscale_dither_image_GPU(PPMPixel* src_img_d, PPMPixel* dst_img_d, int w, int h, int color_depth, int levels);

__host__ __device__
inline uint16_t clamping_add(uint16_t v1, uint16_t v2, uint16_t color_depth) {
    uint32_t sum = uint32_t(v1) + uint32_t(v2);
    return sum > color_depth ? color_depth : sum;
}

__host__ __device__
inline uint16_t clamping_mul(uint16_t v, float s, uint16_t color_depth) {
  uint32_t product = v * s;
  return product > color_depth ? color_depth : product;
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


// NOTE: Device only helpers.

__device__ __forceinline__ float clamp01(float x) {
    return fminf(1.0f, fmaxf(0.0f, x));
}

// Normalize a channel value from the range [0..color_depth] to [0..1].
__device__ __forceinline__ float normalize01(float x, int color_depth) {
    return clamp01(x / (float)color_depth);
}

// Works only on normalized [0..1] values.
// Use normalize01 first if input is in the range [0..color_depth].
__device__ __forceinline__ float srgb_to_linear(float x) {
    return (x <= 0.04045f) ? (x / 12.92f) : __powf((x + 0.055f) / 1.055f, 2.4f);
}

// Expects normalized [0..1] values. Clamps for safety.
__device__ __forceinline__ float linear_to_srgb(float x) {
    x = clamp01(x);
    return (x <= 0.0031308f) ? (12.92f * x) : (1.055f * __powf(x, 1.0f / 2.4f) - 0.055f);
}

// Converts normalized [0..1] sRGB values back to their original range [0..color_depth].
// Intended to be used after linear_to_srgb().
__device__ __forceinline__ int denormalize(float norm_srgb_val, int color_depth) {
    int denorm_val = (int)(norm_srgb_val * (float)color_depth + 0.5f);
    if (denorm_val < 0) denorm_val = 0;
    if (denorm_val > color_depth) denorm_val = color_depth;
    return denorm_val;
}


__device__ __forceinline__ PPMPixel apply_opacity_mask(PPMPixel base_px,
                                                       PPMPixel effect_px,
                                                       float alpha,
                                                       int color_depth)
{
    alpha = clamp01(alpha);
    if (alpha <= 0.0f) return base_px;
    if (alpha >= 1.0f) return effect_px;

    // Normalze both pixels to [0..1]
    float base_r_norm = normalize01((float)base_px.r, color_depth);
    float base_g_norm = normalize01((float)base_px.g, color_depth);
    float base_b_norm = normalize01((float)base_px.b, color_depth);

    float fx_r_norm = normalize01((float)effect_px.r, color_depth);
    float fx_g_norm = normalize01((float)effect_px.g, color_depth);
    float fx_b_norm = normalize01((float)effect_px.b, color_depth);

    // Convert to linear light from sRGB
    float base_r_lin = srgb_to_linear(base_r_norm);
    float base_g_lin = srgb_to_linear(base_g_norm);
    float base_b_lin = srgb_to_linear(base_b_norm);

    float fx_r_lin = srgb_to_linear(fx_r_norm);
    float fx_g_lin = srgb_to_linear(fx_g_norm);
    float fx_b_lin = srgb_to_linear(fx_b_norm);

    // Blend in linear light.
    float inverse_alpha = 1.0f - alpha;
    float out_r_lin = inverse_alpha * base_r_lin + alpha * fx_r_lin;
    float out_g_lin = inverse_alpha * base_g_lin + alpha * fx_g_lin;
    float out_b_lin = inverse_alpha * base_b_lin + alpha * fx_b_lin;

    // Convert back to linear light and de-normalize.
    PPMPixel out_px;
    out_px.r = (uint16_t)denormalize(linear_to_srgb(out_r_lin), color_depth);
    out_px.g = (uint16_t)denormalize(linear_to_srgb(out_g_lin), color_depth);
    out_px.b = (uint16_t)denormalize(linear_to_srgb(out_b_lin), color_depth);
    return out_px;
}