__global__ void apply_opacity_mask_kernel(const PPMPixel* base_img_d,
                                          PPMPixel* effect_img_d,
                                          const float* mask_d,
                                          int w, int h,
                                          int color_depth)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col >= w || row >= h) return;

    int idx = row * w + col;

    float alpha = mask_d ? mask_d[idx] : 1.0f;

    PPMPixel base_px = base_img_d[idx];
    PPMPixel effect_px = effect_img_d[idx];

    effect_inout_d[idx] = apply_opacity_mask(base_px, effect_px, alpha, color_depth);
}
