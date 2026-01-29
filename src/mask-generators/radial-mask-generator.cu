#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include "mask-utils.cuh"
#include "../img-utils.cuh"


__device__ __forceinline__
float ellipse_hit_dist(float dirx, float diry, float rx, float ry)
{
    float denominator = (dirx * dirx) / (rx * rx) + (diry * diry) / (ry * ry);
    return 1.0f / sqrtf(denominator);
}

__global__ void radial_mask_kernel(float* mask_d,
                                   int w, int h,
                                   int inner_percent,
                                   int outer_percent)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col >= w || row >= h) return;

    int idx = row * w + col;

    const float min_tolerance = 1e-6f;

    float inner = inner_percent / 100.0f;
    inner = fmaxf(inner, 0.0f);

    float outer = outer_percent / 100.0f;
    outer = fmaxf(outer, 0.0f);

    // Center of the mask in pixel coords
    float cx = 0.5f * (w - 1);
    float cy = 0.5f * (h - 1);

    // scale_x and scale_y define what a 100% radius means for the ellipses.
    // (The calculation happens to be the same as for cx and cy, but if we moved
    // the center of the mask then that would no longer be the case.)
    float scale_x = 0.5f * (w - 1);
    float scale_y = 0.5f * (h - 1);

    // Radii for inner and outer ellipses
    float rx_in  = fmaxf(inner * scale_x, min_tolerance);
    float ry_in  = fmaxf(inner * scale_y, min_tolerance);

    float rx_out = fmaxf(outer * scale_x, min_tolerance);
    float ry_out = fmaxf(outer * scale_y, min_tolerance);

    // x and y vectors from center to this pixel
    float vx = (float)col - cx;
    float vy = (float)row - cy;

    float dist = sqrtf(vx * vx + vy * vy);

    // The distance is undefined at the center, therefore the center is
    // always treated as though it's inside the inner ellipse.
    if (dist < min_tolerance) {
        mask_d[idx] = 0.0f;
        return;
    }

    // Unit vectors for the direction from the center to this pixel.
    float dirx = vx / dist;
    float diry = vy / dist;

    // Calculate how far a ray that starts at the center of the image,
    // going through this pixel, has to travel before it hits the boundary of
    // the inner and outer ellipse.
    float t_inner = ellipse_hit_dist(dirx, diry, rx_in,  ry_in);
    float t_outer = ellipse_hit_dist(dirx, diry, rx_out, ry_out);

    // For the fade in to work, inner must be less than outer.
    // If it isn't we remove the fade in entirely:
    if (t_outer <= t_inner + min_tolerance) {
        mask_d[idx] = (dist <= t_inner) ? 0.0f : 1.0f;
        return;
    }

    // The pixel is inside inner ellipse, alpha is 0.
    if (dist <= t_inner) {
        mask_d[idx] = 0.0f;
        return;
    }

    // The pixel is outside the outer ellipse, alpha is 1.
    if (dist >= t_outer) {
        mask_d[idx] = 1.0f;
        return;
    }

    // If we reach this point, the pixel is in the transition band between
    // the inner and the outer ellipse. This calculates a linear gradient:
    float u = (dist - t_inner) / (t_outer - t_inner + min_tolerance); // min_tolerance is added to prevent division by zero
    u = clamp01(u);


    // Here it's possible to scale u to make the transition look different.
    // I've tried:
    // 1. alpha = u
    // 2. alpha = u * u * (3.0f - 2.0f * u);
    // 3. alpha = powf(u, 2.0f);

    // A linear gradient (1.) between the inner ellipse and outer ellipse looks
    // good for most effects, but for certain ones with sharp artifacts
    // (like dithering) the edges of the ellipses are noticeable.
    
    // To fix that, (2.) eases the gradient at the edges of the transition.
    // The difference is subtle, it looks very similar to a linear fade for
    // most effects.

    // (3.) is something entirely different. It fades in more slowly and
    // ramps up at the outer edge, making it look more defined.

    // Option 2 seems to work well for most effects:
    float alpha = u * u * (3.0f - 2.0f * u);

    mask_d[idx] = alpha;
}


inline int int_div_rnd_up(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

void generate_radial_mask_GPU(float* mask_d, int w, int h, int inner_percent, int outer_percent)
{
    dim3 threads(20, 20);
    dim3 blocks(int_div_rnd_up(w, threads.x), int_div_rnd_up(h, threads.y));
    radial_mask_kernel<<<blocks, threads>>>(mask_d, w, h, inner_percent, outer_percent);
}
