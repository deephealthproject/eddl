
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
CXXFLAGS = -std=c++11  -march=native -mavx -O3
endif
ifeq ($(OPT),2)
$(info set SSE)
CXXFLAGS = -std=c++11  -march=native -msse -O3
endif
ifeq ($(OPT),3)
$(info set no opt)
CXXFLAGS = -std=c++11  -O3
endif
ifeq ($(OPT),4)
$(info set Debug)
CXXFLAGS =  -std=c++11  -g
endif
LIBFLAGS=-lpthread
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
CXXFLAGS =-std=c++11 -march=native -mavx -O3
endif
ifeq ($(OPT),2)
$(info set SSE)
CXXFLAGS =-std=c++11 -march=native -msse -O3
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

ifeq ($(TARGET),1)
GPU_CXXFLAGS := -std=c++11 -I $(cuda_path)include/ -L  $(cuda_path)lib64/ -DcGPU
endif
ifeq ($(TARGET),2)
GPU_CXXFLAGS := -std=c++11 -I $(cuda_path)include/ -L  $(cuda_path)lib/ -DcGPU
endif

GPU_LIBFLAGS := -lcudart -lcublas -lcurand
GPU_CXX := $(cuda_path)bin/nvcc
GPU_OBJ = tensor_cuda.o tensor_cuda_op.o tensor_kernels.o
endif


endif
endif


#######################################################################
OBJ =	utils.o cpu_convol.o tensor.o tensor_op.o layer.o layer_tensor.o layer_input.o layer_dense.o layer_activation.o layer_reshape.o layer_conv.o layer_drop.o layer_add.o layer_cat.o loss.o metric.o optim.o net.o eddl.o

ifeq ($(CORE),1)
all: $(GPU_OBJ) $(OBJ) test_eddl test_tensors
endif

ifeq ($(CORE),2)
all: $(GPU_OBJ) $(OBJ) test_eddl test_eddl_gpu test_tensors
endif

#--------------------------------- c code
utils.o: ../utils.cpp ../utils.h
	$(CXX) $(CXXFLAGS) -c ../utils.cpp

cpu_convol.o: ../cpu/cpu_convol.cpp ../cpu/cpu_convol.h
	$(CXX) $(CXXFLAGS) -c ../cpu/cpu_convol.cpp

tensor.o: ../tensor.cpp ../tensor.h
	$(CXX) $(CXXFLAGS) -c ../tensor.cpp

tensor_op.o: ../tensor_op.cpp ../tensor.h
		$(CXX) $(CXXFLAGS) -c ../tensor_op.cpp

tensor_over.o: ../tensor_over.cpp ../tensor_over.h
		$(CXX) $(CXXFLAGS) -c ../tensor_over.cpp


layer.o: ../layers/layer.cpp ../layers/layer.h
		$(CXX) $(CXXFLAGS) -c ../layers/layer.cpp

layer_tensor.o: ../layers/layer_tensor.cpp ../layers/layer.h
				$(CXX) $(CXXFLAGS) -c ../layers/layer_tensor.cpp

layer_input.o: ../layers/layer_input.cpp ../layers/layer.h
		$(CXX) $(CXXFLAGS) -c ../layers/layer_input.cpp

layer_dense.o: ../layers/layer_dense.cpp ../layers/layer.h
		$(CXX) $(CXXFLAGS) -c ../layers/layer_dense.cpp

layer_activation.o: ../layers/layer_activation.cpp ../layers/layer.h
		$(CXX) $(CXXFLAGS) -c ../layers/layer_activation.cpp

layer_add.o: ../layers/layer_add.cpp ../layers/layer.h
		$(CXX) $(CXXFLAGS) -c ../layers/layer_add.cpp

layer_cat.o: ../layers/layer_cat.cpp ../layers/layer.h
		$(CXX) $(CXXFLAGS) -c ../layers/layer_cat.cpp

layer_conv.o: ../layers/layer_conv.cpp ../layers/layer.h
		$(CXX) $(CXXFLAGS) -c ../layers/layer_conv.cpp
		
layer_drop.o: ../layers/layer_drop.cpp ../layers/layer.h
		$(CXX) $(CXXFLAGS) -c ../layers/layer_drop.cpp

loss.o: ../loss.cpp ../loss.h
		$(CXX) $(CXXFLAGS) -c ../loss.cpp

metric.o: ../metric.cpp ../metric.h
		$(CXX) $(CXXFLAGS) -c ../metric.cpp

optim.o: ../optim.cpp ../optim.h
		$(CXX) $(CXXFLAGS) -c ../optim.cpp

net.o: ../net.cpp ../net.h
		$(CXX) $(CXXFLAGS) -c ../net.cpp

eddl.o: ../eddl.cpp ../eddl.h
		$(CXX) $(CXXFLAGS) -c ../eddl.cpp

layer_reshape.o: ../layers/layer_reshape.cpp ../layers/layer.h
		$(CXX) $(CXXFLAGS) -c ../layers/layer_reshape.cpp

##ADD_MORE_RULES

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

test_tensors: test_tensors.cpp $(OBJ) $(GPU_OBJ)
		$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) test_tensors.cpp -o test_tensors $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

#-------------------------

install:
	cp test_eddl /usr/local/bin
clean:
	rm *.o





###########################################################################
