#include <cstring>
#include <cstdio>
#include <string>
#include "img-utils.cuh"

void run_blur_kernel_test(std::string outfile, BlurType blur_type);

int main(int argc, char** argv) {
    BlurType blur = BLUR_BOX;
    std::string outfile = "";

    if (argc == 2) {
        if (strcmp(argv[1], "box") == 0) {
            blur = BLUR_BOX;
        } else if (strcmp(argv[1], "gauss") == 0 || strcmp(argv[1], "gaussian") == 0) {
            blur = BLUR_GAUSSIAN;
        } else if (strcmp(argv[1], "motion") == 0) {
            blur = BLUR_MOTION;
        } else {
            printf("Unknown blur type '%s'. Use: box | gauss | motion\n", argv[1]);
            return 1;
        }
    } else {
        printf("No blur type given, defaulting to 'box'.\n");
        printf("Usage: %s [box|gauss|motion] < input.ppm\n", argv[0]);
    }
    if(argc == 3) {
        outfile = std::string(argv[2]);
    }

    run_blur_kernel_test(outfile, blur);
    return 0;
}
