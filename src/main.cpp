#include <cstring>
#include <cstdio>
#include <string>
#include "common.hpp"


int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <input.ppm> <output.ppm> <shader> [mask]\n", argv[0]);
        printf("Valid shader names are:\n");
        printf("[ box | gauss | motion |dither | grayscale | sobel | inversion | mirror | cpu-convolution ]\n");
        return 1;
    }

    std::string infile  = argv[1];
    std::string outfile = argv[2];

    ShaderType shader = ShaderType::BLUR_BOX;
    
    if (strcmp(argv[3], "box") == 0) {
        shader = ShaderType::BLUR_BOX;
    } else if (strcmp(argv[3], "gauss") == 0 || strcmp(argv[3], "gaussian") == 0) {
        shader = ShaderType::BLUR_GAUSSIAN;
    } else if (strcmp(argv[3], "motion") == 0) {
        shader = ShaderType::BLUR_MOTION;
    } else if (strcmp(argv[3], "dither") == 0) {
        shader = ShaderType::DITHER;
    } else if (strcmp(argv[3], "grayscale") == 0) {
        shader = ShaderType::GRAYSCALE;
    } else if (strcmp(argv[3], "sobel") == 0) {
        shader = ShaderType::BLUR_SOBEL;
    } else if (strcmp(argv[3], "invert") == 0 || strcmp(argv[3], "inversion") == 0) {
        shader = ShaderType::INVERSION;
    } else if (strcmp(argv[3], "mirror") == 0) {
        shader = ShaderType::MIRROR;
    } else if (strcmp(argv[3], "cpu-convolution") == 0) {
        // CPU-only path, bypass CUDA setup; handled below.
        // Run CPU convolution directly and exit.
        extern void run_cpu_convolution(std::string infile, std::string outfile);
        run_cpu_convolution(infile, outfile);
        return 0;
    } else {
        printf("Unknown shader '%s'. Valid shader names are:\n", argv[3]);
        printf("[ box | gauss | motion | dither |grayscale | sobel | inversion | mirror | cpu-convolution ]\n");
        return 1;
    }

    MaskSpec mask;

    if (argc >= 5) {
        std::string m = argv[4];
        if (m.size() >= 4 && m.substr(m.size() - 4) == ".ppm") {
            mask.type = MaskType::PPM;
            mask.path = m;
        } else if (m == "gradient") {
            mask.type = MaskType::GRADIENT;
        } else if (m == "radial") {
            mask.type = MaskType::RADIAL;
        } else if (m == "none") {
            mask.type = MaskType::NONE;
        } else {
            printf("Unknown mask '%s'. Valid masks are: [ none | gradient | radial | <file>.ppm ]\n", m.c_str());
            return 1;
        }
    }

    run_kernel(infile, outfile, shader, mask);
    return 0;
}
