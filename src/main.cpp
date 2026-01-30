#include <cstring>
#include <cstdio>
#include <string>
#include "common.hpp"


bool starts_with(const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <input.ppm> <output.ppm> <shader> [mask]\n", argv[0]);
        printf("Valid shader names are:\n");
        printf("[ box | gauss | motion |dither | grayscale | sobel | inversion | mirror | cpu-convolution ]\n");
        return 1;
    }

    std::string infile  = argv[1];
    std::string outfile = argv[2];
    std::string shader_arg = argv[3];

    ShaderSpec shader;
    
    if (shader_arg == "box") {
        shader.type = ShaderType::BLUR_BOX;
    } else if (shader_arg == "gauss" || shader_arg == "gaussian") {
        shader.type = ShaderType::BLUR_GAUSSIAN;
    } else if (shader_arg == "motion") {
        shader.type = ShaderType::BLUR_MOTION;
    } else if (starts_with(shader_arg, "dither")) {
        shader.type = ShaderType::DITHER;

        size_t d1 = shader_arg.find('-');
        if (d1 != std::string::npos) {
            if (shader_arg.find('-', d1 + 1) != std::string::npos) {
                printf("Invalid shader '%s': dither accepts at most one '-'. Use dither-<int>\n",
                    shader_arg.c_str());
                return 1;
            }

            std::string level_str = shader_arg.substr(d1 + 1);
            try {
                shader.dither_levels = std::stoi(level_str);
            } catch (...) {
                printf("Invalid shader '%s': expected an int after '-'.\n",
                    shader_arg.c_str());
                return 1;
            }
        }
    } else if (shader_arg == "grayscale") {
        shader.type = ShaderType::GRAYSCALE;
    } else if (shader_arg == "sobel") {
        shader.type = ShaderType::BLUR_SOBEL;
    } else if (shader_arg == "invert" || shader_arg == "inversion") {
        shader.type = ShaderType::INVERSION;
    } else if (shader_arg == "mirror") {
        shader.type = ShaderType::MIRROR;
    } else if (shader_arg == "cpu-convolution") {
        // CPU-only path, bypass CUDA setup; handled below.
        // Run CPU convolution directly and exit.
        extern void run_cpu_convolution(std::string infile, std::string outfile);
        run_cpu_convolution(infile, outfile);
        return 0;
    } else {
        printf("Unknown shader '%s'. Valid shader names are:\n", shader_arg.c_str());
        printf("[ box | gauss | motion | dither |grayscale | sobel | inversion | mirror | cpu-convolution ]\n");
        return 1;
    }

    MaskSpec mask;

    if (argc >= 5) {
        std::string m = argv[4];
        if (m.size() >= 4 && m.substr(m.size() - 4) == ".ppm") {
            mask.type = MaskType::PPM;
            mask.path = m;
        } else if (starts_with(m, "gradient")) {
            mask.type = MaskType::GRADIENT;

            // Check for first dash
            size_t d1 = m.find('-');
            if (d1 == std::string::npos) {
                // No dashes, keep defaults
            } else {
                // Check for a second dash (not allowed)
                if (m.find('-', d1 + 1) != std::string::npos) {
                    printf("Invalid mask '%s': gradient accepts at most one '-'.\n", m.c_str());
                    return 1;
                }

                std::string dir_str = m.substr(d1 + 1);

                if (dir_str == "right") {
                    mask.gradient_dir = GradientDirection::LEFT_TO_RIGHT;
                } else if (dir_str == "left") {
                    mask.gradient_dir = GradientDirection::RIGHT_TO_LEFT;
                } else if (dir_str == "bottom") {
                    mask.gradient_dir = GradientDirection::TOP_TO_BOTTOM;
                } else if (dir_str == "top") {
                    mask.gradient_dir = GradientDirection::BOTTOM_TO_TOP;
                } else {
                    printf("Invalid mask '%s': unknown gradient direction '%s'.\n", m.c_str(), dir_str.c_str());
                    printf("Use gradient-<left|right|top|bottom>.\n");
                    return 1;
                }
            }
        } else if (starts_with(m, "radial")) {
            mask.type = MaskType::RADIAL;

            // Check for a first dash
            size_t d1 = m.find('-');
            if (d1 == std::string::npos) {
                // No dashes, keep defaults
            } else {
                size_t d2 = m.find('-', d1 + 1);
                if (d2 == std::string::npos) {
                    // radial-<inner>
                    std::string inner_str = m.substr(d1 + 1);
                    try {
                        mask.radial_inner = std::stoi(inner_str);
                    } catch (...) {
                        printf("Invalid mask '%s': expected an int after '-'.\n",
                            m.c_str());
                        return 1;
                    }
                } else {
                    // Check for a third dash (not allowed)
                    if (m.find('-', d2 + 1) != std::string::npos) {
                        printf("Invalid mask '%s': radial accepts at most two '-'.\n", m.c_str());
                        printf("Use radial-<int>-<int> or radail-<int>.\n");
                        return 1;
                    }

                    // radial-<inner>-<outer>
                    std::string inner_str = m.substr(d1 + 1, d2 - (d1 + 1));
                    std::string outer_str = m.substr(d2 + 1);

                    try {
                        mask.radial_inner = std::stoi(inner_str);
                    } catch (...) {
                        printf("Invalid mask '%s': expected an int after first '-'.\n", m.c_str());
                        return 1;
                    }

                    try {
                        mask.radial_outer = std::stoi(outer_str);
                    } catch (...) {
                        printf("Invalid mask '%s': expected an int after second '-'.\n", m.c_str());
                        return 1;
                    }
                }
            }
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
