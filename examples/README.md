# Shader Usage Examples

The following table demonstrates the output of various shader and mask combinations applied to the input image.

| Command                                                               | Shader          | Mask     | Result                                  |
| :-------------------------------------------------------------------- | :-------------- | :------- | :-------------------------------------- |
| _Original Input File_                                                 | **Input Image** | N/A      | ![Input](input.png)                     |
| `./build/main examples/input.ppm examples/output.ppm box`             | Box             | None     | ![Box](box.png)                         |
| `./build/main examples/input.ppm examples/output.ppm gauss`           | Gauss           | None     | ![Gauss](gauss.png)                     |
| `./build/main examples/input.ppm examples/output.ppm motion`          | Motion          | None     | ![Motion](motion.png)                   |
| `./build/main examples/input.ppm examples/output.ppm dither-10`       | Dither-10       | None     | ![Dither 10](dither_10.png)             |
| `./build/main examples/input.ppm examples/output.ppm dither-60`       | Dither-60       | None     | ![Dither 60](dither_60.png)             |
| `./build/main examples/input.ppm examples/output.ppm grayscale`       | Grayscale       | None     | ![Grayscale](grayscale.png)             |
| `./build/main examples/input.ppm examples/output.ppm sobel`           | Sobel           | None     | ![Sobel](sobel.png)                     |
| `./build/main examples/input.ppm examples/output.ppm inversion`       | Inversion       | None     | ![Inversion](inversion.png)             |
| `./build/main examples/input.ppm examples/output.ppm mirror`          | Mirror          | None     | ![Mirror](mirror.png)                   |
| `./build/main examples/input.ppm examples/output.ppm cpu-convolution` | CPU Convolution | None     | ![CPU Convolution](cpu_convolution.png) |
| `./build/main examples/input.ppm examples/output.ppm sobel radial`    | Sobel           | Radial   | ![Sobel Radial](sobel_radial.png)       |
| `./build/main examples/input.ppm examples/output.ppm sobel gradient`  | Sobel           | Gradient | ![Sobel Gradient](sobel_gradient.png)   |
