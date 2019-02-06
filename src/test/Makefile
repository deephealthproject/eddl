
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),install)

#DEFAULT VALUES
## TARGET=1 LINUX
ifdef TARGET
ifeq ($(TARGET),1)
$(info set Linux)
endif
ifeq ($(TARGET),2)
$(info set Mac)
endif
ifeq ($(TARGET),3)
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


## CPU OPTIMIZATIONS
ifndef OPT
OPT=3
$(info ----> DEFAULT: no OPT)
endif



#### LINUX
ifeq ($(TARGET),1)
# OPT=1: AVX
# OPT=2: SSE
# OPT=3: Nothing
# OPT=4: Debug
CXX = g++
ifeq ($(OPT),1)
$(info set AVX)
CXXFLAGS = -std=c++11 -fopenmp -march=native -mavx -O3
endif
ifeq ($(OPT),2)
$(info set SSE)
CXXFLAGS = -std=c++11 -fopenmp -march=native -msse2 -O3
endif
ifeq ($(OPT),3)
$(info set no opt)
CXXFLAGS = -std=c++11 -fopenmp -O3
endif
ifeq ($(OPT),4)
$(info set Debug)
CXXFLAGS =  -std=c++11 -fopenmp -g
endif
endif


#### MAC
ifeq ($(TARGET),2)
# OPT=1: AVX
# OPT=2: SSE
# OPT=3: Nothing
# OPT=4: Debug
CXX = g++
ifeq ($(OPT),1)
$(info set AVX)
CXXFLAGS =-std=c++14 -march=native -mavx -O3
endif
ifeq ($(OPT),2)
$(info set SSE)
CXXFLAGS =-std=c++11-march=native -msse2 -O3
endif
ifeq ($(OPT),3)
$(info set no opt)
CXXFLAGS =-std=c++11 -O3
endif
ifeq ($(OPT),4)
$(info set Debug)
CXXFLAGS =-std=c++11 -g
endif
endif



##CORE FLAGS
ifeq ($(CORE),2)
cuda_path := $(CUDA_PATH)
ifeq ($(cuda_path),)
$(error Set enviroment variable CUDA_PATH with directory to cuda)
endif

CXXFLAGS :=  $(CXXFLAGS) -DcGPU -I $(cuda_path)include/
GPU_CXXFLAGS := -std=c++11 -I $(cuda_path)include/ -L  $(cuda_path)lib64/ -DcGPU
GPU_LIBFLAGS := -lcudart -lcublas -lcurand
GPU_CXX := $(cuda_path)bin/nvcc
GPU_OBJ = tensor_cuda.o tensor_cuda_op.o tensor_kernels.o
endif


endif
endif


#######################################################################
OBJ =	tensor.o tensor_op.o layer.o layer_input.o layer_dense.o layer_activation.o layer_add.o optim.o net.o

ifeq ($(CORE),1)
all: $(GPU_OBJ) $(OBJ) test_eddl
endif

ifeq ($(CORE),2)
all: $(GPU_OBJ) $(OBJ) test_eddl_gpu
endif

#--------------------------------- c code
tensor.o: ../tensor.cpp ../tensor.h
	$(CXX) $(CXXFLAGS) -c ../tensor.cpp

tensor_op.o: ../tensor_op.cpp ../tensor.h
		$(CXX) $(CXXFLAGS) -c ../tensor_op.cpp

tensor_over.o: ../tensor_over.cpp ../tensor_over.h
		$(CXX) $(CXXFLAGS) -c ../tensor_over.cpp


layer.o: ../layer.cpp ../layer.h
		$(CXX) $(CXXFLAGS) -c ../layer.cpp

layer_input.o: ../layer_input.cpp ../layer.h
		$(CXX) $(CXXFLAGS) -c ../layer_input.cpp

layer_dense.o: ../layer_dense.cpp ../layer.h
		$(CXX) $(CXXFLAGS) -c ../layer_dense.cpp

layer_activation.o: ../layer_activation.cpp ../layer.h
		$(CXX) $(CXXFLAGS) -c ../layer_activation.cpp

layer_add.o: ../layer_add.cpp ../layer.h
		$(CXX) $(CXXFLAGS) -c ../layer_add.cpp

optim.o: ../optim.cpp ../optim.h
		$(CXX) $(CXXFLAGS) -c ../optim.cpp

net.o: ../net.cpp ../net.h
		$(CXX) $(CXXFLAGS) -c ../net.cpp

#---------------------------------- gpu code
tensor_cuda.o: ../gpu/tensor_cuda.cu ../gpu/tensor_cuda.h
		$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c ../gpu/tensor_cuda.cu

tensor_cuda_op.o: ../gpu/tensor_cuda_op.cu ../gpu/tensor_cuda_op.h
		$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c ../gpu/tensor_cuda_op.cu

tensor_kernels.o: ../gpu/tensor_kernels.cu ../gpu/tensor_kernels.h
		$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c ../gpu/tensor_kernels.cu

#---------------------------------- release
test_eddl: test_eddl.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) test_eddl.cpp -o test_eddl $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

test_eddl_gpu: test_eddl_gpu.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) test_eddl_gpu.cpp -o test_eddl_gpu $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

#-------------------------

install:
	cp test_eddl /usr/local/bin
clean:
	rm *.o





###########################################################################
