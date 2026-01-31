# GPU Performance Analysis & Optimization Report

The primary bottleneck is **Uncoalesced Global Memory Access** caused by the alignment of the pixel data structure.

### 1. Memory Access Inefficiency

- **Issue:** The PPMPixel structure is 6 bytes (3 channels × 2 bytes). Because this size is not a power of 2, threads within a warp access memory addresses that are not aligned to 32-byte sectors.

- **Example Report** “Uncoalesced global access, expected 1136520 sectors, got 3409560 (3.00x) at PC 0x7fff12263b10”

- **Possible solution**
  Fix this using padding to align to a structure to the power of 2

```c
typedef struct {
    uint16_t r;
    uint16_t g;
    uint16_t b;
    uint16_t padding;  // Padding added to align the structure to 8 bytes
} PPMPixel;
```

Should reduce the sectors per request from 3.00x to 1.00x if correctly implemented.

Also switching our array structure from AoS to SoA should be a viable solution but that would mean a restructuration of the pipeline.

One rather obvious fix would be to use less memory for the channel values, as sRGB channels typically use 8 bits. The reason we stuck with 16 bits is that the PPM file format can technically be configured to use 16 bits, even if it’s uncommon. We thought that technical correctness makes more sense in this case.

### 2. Warp stalls (Latency)

- **Issue:** There is high amount of time when the warps are idle, in this case about ⅓ of the their execution time they are unable to process the instruction because they are waiting for data.

- **Example Report** “On average, each warp of this kernel spends 9.6 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. This represents about 35.8% of the total average of 26.7 cycles between issuing two instructions. To reduce the number of cycles waiting on L1TEX data accesses verify the memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by increasing data locality or by changing the cache configuration, and consider moving frequently used data to shared memory.”

- **Possible solution** Same as for memory access inefficiency, should be significantly reduced if we fix the bandwidth issue as it makes fewer L1TEX required per warp.

### 3. Convolution Based Shaders

The convolution based shader: **box blur,gaussian blur, and motion** all as expected have the same compute throughput and memory throughput as they use the same kernel. Compute throughput being at ~97,6% and memory throughput ~70,8%. The kernel is compute bound but as they are all using tiling and shared memory it is not memory bound.
XU is the highest-utilized pipeline at 97.7%. The second most utilized pipeline, LSU, is only at 31.48%. We can conclude that the XU pipeline is over-utilized and a performance bottleneck.

### 4. Other Shaders

The other shaders work only on single pixels, which means that unlike the convolution kernel they can’t make use of tiling to make them less memory bound. Some of them are still compute bound though, since the calculations are so involved. The most memory bound kernels are the inversion kernel and the mirror kernel. Both of them have ~17% compute throughput and ~80% memory throughput.

The grayscale kernel is running a much more complicated calculation since it has to convert the sRGB channel values to linear light before applying the Luma formula. The sRGB to linear light calculation is quite heavy and is the reason that the compute throughput is at ~80% while the memory throughput is at ~63%. The XU pipeline is well utilized at ~70%.

The grayscale dithering kernel starts off as the regular grayscale kernel, but adds additional calculations at the end. Therefore it’s the most compute bound shader kernel apart from the convolutional shader kernels. It has ~86% compute throughput and only 50.82% memory throughput. It uses a mix of pipelines with the XU pipeline being the most used at 73.62%, but the FMA, ADU, and ALU pipelines are all utilized at around 50%. The linear light conversion is still probably the heaviest calculation for the dithering kernel, and both the regular grayscale kernel and the dithering kernel could probably be optimized by using a faster approximation rather than the exact formula. The differences in the output would likely be imperceptible.

### 5. Masks

The masks, like the other shaders, work on single pixel values. Like the dithering kernel, their computations are rather complex, which is why they’re not outright memory bound. The gradient mask kernel is memory bound with 96.65% memory throughput, but it also manages a compute throughput of 69.06%. The radial mask kernel is highly compute bound at 86.11% compute throughput, and only 20.75% memory throughput. Of all the kernels that work on single pixels, it does by far the most complex calculation, so it makes sense.

The XU pipeline is the most used pipeline for both mask kernels, and on the radial kernel it’s likely a bottleneck at 86.20%. Since the gradient kernel is memory bound, the XU pipeline isn’t quite as stressed at only 63.38%, but it could potentially become a bottleneck if there was some way to make it less memory bound. Both masks also make good use of the FMA pipeline, the throughput for the radial mask kernel is at 75.05% while the gradient mask kernel is at 51.59%.
