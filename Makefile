# Template for Makefile from: https://www.geeksforgeeks.org/cpp/makefile-in-c-and-its-applications/

# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -Wall -g -O3
NVCC_FLAGS = -g -O3

# Target executable
BUILD_DIR = build
TARGET = $(BUILD_DIR)/main

# Source files
SRCS = src/main.cpp src/img.cpp src/convolution-cpu.cpp
HEADERS = src/img.hpp
CU_SRCS = src/setup.cu src/convolution-kernel.cu src/grayscale-kernel.cu src/grayscale-dither-kernel.cu src/mask-generators/gradient-mask-generator.cu src/inversion-kernel.cu src/mirror-kernel.cu

# Object files
OBJS := $(addprefix $(BUILD_DIR)/, $(SRCS:.cpp=.o))
CU_OBJS := $(addprefix $(BUILD_DIR)/, $(CU_SRCS:.cu=.o))

# Default rule to build and run the executable
all: $(TARGET)

# Rule to link object files into the target executable
$(TARGET): $(OBJS) $(CU_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Make sure the build directory is present
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean rule to remove generated files
clean:
	rm -rf $(BUILD_DIR)
