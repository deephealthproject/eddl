
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),install)

#DEFAULT VALUES
## TARGET=1 LINUX
ifdef TARGET

ifeq ($(TARGET),1)
$(info set Linux)
endif

ifeq ($(TARGET),2)
$(info set Windows)
endif
else
TARGET=1
$(info ----> DEFAULT: set TARGET LINUX)
endif


## CORE=0 CPU
ifdef CORE
ifeq ($(CORE),1)
$(info set CPU)
endif
ifeq ($(CORE),2)
$(info set GPU)
endif
ifeq ($(CORE),3)
$(info set FPGA)
endif
else
CORE=1
$(info ----> DEFAULT: set CORE CPU)
endif



#### LINUX
ifeq ($(TARGET),1)
# OPT=1: AVX
# OPT=2: SSE
# OPT=3: Nothing
# OPT=4: Debug
CXX = g++
ifdef OPT
ifeq ($(OPT),1)
$(info set AVX)
CXXFLAGS = -fopenmp -I Eigen -march=native -mavx -O3
endif
ifeq ($(OPT),2)
$(info set SSE)
CXXFLAGS = -fopenmp -I Eigen -march=native -msse2 -O3
endif
ifeq ($(OPT),3)
$(info set no opt)
CXXFLAGS = -fopenmp -I Eigen -O3
endif
ifeq ($(OPT),4)
$(info set Debug)
CXXFLAGS =  -fopenmp -I Eigen -g
endif
endif
endif




##CORE FLAGS
ifeq ($(CORE),2)
cuda_path := $(CUDA_PATH)
ifeq ($(cuda_path),)
$(error Set enviroment variable CUDA_PATH with directory to cuda)
endif

CXXFLAGS :=  $(CXXFLAGS) -DuseGPU -I $(cuda_path)include/
GPU_CXXFLAGS := -I $(cuda_path)include/ -L  $(cuda_path)lib64/ -DuseGPU
GPU_LIBFLAGS := -lcudart -lcublas -lcurand
GPU_CXX := $(cuda_path)bin/nvcc
GPU_OBJ = tensor_cuda.o
endif


endif
endif


#######################################################################
OBJ =	tensor.o

all: $(GPU_OBJ) $(OBJ) test_eddl


#--------------------------------- c code
tensor.o: ../tensor.cpp ../tensor.h
	$(CXX) $(CXXFLAGS) -c ../tensor.cpp

#---------------------------------- gpu code
tensor_cuda.o: ../gpu/tensor_cuda.cu ../gpu/tensor_cuda.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c ../gpu/tensor_cuda.cu

#---------------------------------- release
test_eddl: test_eddl.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) test_eddl.cpp -o test_eddl $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)


#-------------------------

install:
	cp test_eddl /usr/local/bin
clean:
	rm *.o



###########################################################################
