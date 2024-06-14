# Product Names
TARGET = gpu_demo

# Input Names
CUDA_SRC = gpu_regex.cu
CPP_SRC = gpu_demo.cpp
NFA_SRC = nfa.cpp

# Object files
CUDA_OBJ = gpu_regex.o
CPP_OBJ = gpu_demo.o
NFA_OBJ = nfa.o

# CUDA Compiler and Flags
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc
NVCC_COMPILE_FLAGS = -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
             --expt-relaxed-constexpr -I$(CUDA_INC_PATH) -gencode arch=compute_52,code=sm_52
NVCC_LINK_FLAGS = -g -Wno-deprecated-gpu-targets --std=c++11 -I$(CUDA_INC_PATH)
NVCC_LIBS = -L$(CUDA_LIB_PATH) -lcudart

# C++ Compiler and Flags
GPP = g++
CPP_FLAGS = -g -Wall -std=c++11 -pthread -I$(CUDA_INC_PATH)
CPP_LIBS = -L$(CUDA_LIB_PATH) -lcudart

# ------------------------------------------------------------------------------
# Make Rules
# ------------------------------------------------------------------------------

# Top level rule
all: $(TARGET)

$(TARGET): $(CUDA_OBJ) $(CPP_OBJ) $(NFA_OBJ)
	$(NVCC) $(NVCC_LINK_FLAGS) $^ -o $@ $(CPP_LIBS)

# Compile CUDA Source Files into an object file
$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCC_COMPILE_FLAGS) -c $< -o $@

# Compile C++ Source Files into an object file
$(CPP_OBJ): $(CPP_SRC)
	$(GPP) $(CPP_FLAGS) -c $< -o $@

$(NFA_OBJ): $(NFA_SRC)
	$(GPP) $(CPP_FLAGS) -c $< -o $@

# Clean everything including temporary Emacs files
clean:
	rm -f $(TARGET) *.o *~
	rm -f src/*~

.PHONY: clean
