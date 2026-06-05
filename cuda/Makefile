
# Compiler setup


# nvcc compiles .cu files (CUDA + C++ hybrid)
# gcc compiles .c files (plain C)
# nvcc also acts as the linker — it knows where CUDA runtime libs live
NVCC     = nvcc
CC       = gcc

# -G = debug symbols for GPU code (remove for release builds)
# -g = debug symbols for CPU code
# -lcublas = link against cuBLAS library
NVFLAGS  = -g -lcublas
CFLAGS   = -g -Wall -Wextra

# Source files — organized by layer

# Layer 0: CUDA runtime + cuBLAS handle + device memory
CUDA_BACKEND = cuda/cuda_backend.cu cuda/cuda_mem.cu

# Layer 1: GPU compute kernels
CUDA_KERNELS = cuda/kernels/elementwise.cu \
               cuda/kernels/reduce.cu      \
               cuda/kernels/matmul.cu      \
               cuda/kernels/conv2d.cu      \
               cuda/kernels/batchnorm.cu

# All CUDA sources
CU_SRCS = $(CUDA_BACKEND) $(CUDA_KERNELS)

# Layer 2+: plain C files (add these as we build them)
# C_SRCS = tensor/tensor.c tensor/tensor_ops.c ...
# Object files — swap .cu/.c extension for .o
CU_OBJS = $(CU_SRCS:.cu=.o)
# C_OBJS  = $(C_SRCS:.c=.o)

# Build rules

# Final binary nvcc links everything because it resolves
# CUDA runtime and cuBLAS automatically
all: main

main: main.o $(CU_OBJS)
	$(NVCC) -o $@ $^ $(NVFLAGS)

# .cu files → nvcc
# The -I flags let kernels find headers in parent cuda/ directory
%.o: %.cu
	$(NVCC) -c $< -o $@ $(NVFLAGS) -Icuda

# .c files → gcc
# -x c forces C mode (not C++)
%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

# main.c is plain C but needs to link with CUDA,
# so we compile it with gcc and let nvcc handle linking above
main.o: main.c
	$(CC) -c $< -o $@ $(CFLAGS)


# Convenience targets

clean:
	del /S /Q *.o *.exe 2>nul || true

# Build just Layer 0 + 1 to verify CUDA compiles
test-cuda: $(CU_OBJS)
	@echo CUDA layers compiled OK

.PHONY: all clean test-cuda