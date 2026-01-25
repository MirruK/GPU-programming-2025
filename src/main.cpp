#include <cstring>
#include <cstdio>
#include <string>
#include "common.hpp"


int main(int argc, char** argv) {
    ShaderType shader = ShaderType::BLUR_BOX;
    std::string outfile = "";

    if (argc >= 2) {
        if (strcmp(argv[1], "box") == 0) {
            shader = ShaderType::BLUR_BOX;
        } else if (strcmp(argv[1], "gauss") == 0 || strcmp(argv[1], "gaussian") == 0) {
            shader = ShaderType::BLUR_GAUSSIAN;
        } else if (strcmp(argv[1], "motion") == 0) {
            shader = ShaderType::BLUR_MOTION;
        } else if (strcmp(argv[1], "dither") == 0) {
            shader = ShaderType::DITHER;
        } else if (strcmp(argv[1], "grayscale") == 0) {
            shader = ShaderType::GRAYSCALE;
        } else if (strcmp(argv[1], "sobel") == 0) {
            shader = ShaderType::BLUR_SOBEL;
        } else {
            printf("Unknown shader '%s'. Use: box | gauss | motion | dither | grayscale | sobel\n", argv[1]);
            return 1;
        }
    } else {
        printf("No shader type given, defaulting to 'box'.\n");
        printf("Usage: %s [box|gauss|motion|dither|grayscale|sobel] < input.ppm\n", argv[0]);
    }
    if(argc == 3) {
        outfile = std::string(argv[2]);
    }

    run_kernel(outfile, shader);
    return 0;
}
