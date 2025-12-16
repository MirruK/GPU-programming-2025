# Template for Makefile from: https://www.geeksforgeeks.org/cpp/makefile-in-c-and-its-applications/

# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -Wall -g -O3

# Target executable
BUILD_DIR = build
TARGET = $(BUILD_DIR)/main

# Source files
SRCS = src/main.cpp src/img.cpp
HEADERS = src/img.hpp
CU_SRCS = src/setup.cu src/blur-kernel.cu #src/main.cu

# Object files
OBJS := $(addprefix $(BUILD_DIR)/, $(notdir $(SRCS:.cpp=.o)))
CU_OBJS := $(addprefix $(BUILD_DIR)/, $(notdir $(CU_SRCS:.cu=.o)))

# Default rule to build and run the executable
all: $(TARGET)

# Rule to link object files into the target executable
$(TARGET): $(OBJS) $(CU_OBJS)
	$(NVCC) -o $@ $^

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Make sure the build directory is present
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean rule to remove generated files
clean:
	rm -r $(BUILD_DIR)
