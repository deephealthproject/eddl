
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


## CORE=1 CPU
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

SRC_PATH=../../../src

#######################################################################
OBJ =	utils.o tensor.o tensor_op.o cpu_convol.o compserv.o layer.o layer_tensor.o layer_transpose.o layer_input.o layer_embedding.o layer_dense.o layer_conv.o layer_pool.o layer_activation.o layer_reshape.o layer_drop.o layer_add.o layer_cat.o layer_average.o layer_matmul.o layer_maximum.o layer_minimum.o layer_subtract.o loss.o metric.o optim.o net.o eddl.o wrapper.o

ifeq ($(CORE),1)  # CPU
all: $(GPU_OBJ) $(OBJ) test_eddl test_wrapper
endif

ifeq ($(CORE),2)  # GPU
all: $(GPU_OBJ) $(OBJ) test_eddl test_wrapper
endif

#--------------------------------- c code
utils.o: $(SRC_PATH)/utils.cpp $(SRC_PATH)/utils.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/utils.cpp

cpu_convol.o: $(SRC_PATH)/cpu/cpu_convol.cpp $(SRC_PATH)/cpu/cpu_convol.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/cpu/cpu_convol.cpp

tensor.o: $(SRC_PATH)/tensor.cpp $(SRC_PATH)/tensor.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor.cpp

tensor_op.o: $(SRC_PATH)/tensor_op.cpp $(SRC_PATH)/tensor.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor_op.cpp

tensor_over.o: $(SRC_PATH)/tensor_over.cpp $(SRC_PATH)/tensor_over.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor_over.cpp

compserv.o: $(SRC_PATH)/compserv.cpp $(SRC_PATH)/compserv.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/compserv.cpp

layer.o: $(SRC_PATH)/layers/layer.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer.cpp

layer_tensor.o: $(SRC_PATH)/layers/layer_tensor.cpp $(SRC_PATH)/layers/layer.h
				$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_tensor.cpp

layer_transpose.o: $(SRC_PATH)/layers/layer_transpose.cpp $(SRC_PATH)/layers/layer.h
				$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_transpose.cpp

layer_input.o: $(SRC_PATH)/layers/layer_input.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_input.cpp

layer_embedding.o: $(SRC_PATH)/layers/layer_embedding.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_embedding.cpp

layer_dense.o: $(SRC_PATH)/layers/layer_dense.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_dense.cpp

layer_conv.o: $(SRC_PATH)/layers/layer_conv.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_conv.cpp

layer_pool.o: $(SRC_PATH)/layers/layer_pool.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_pool.cpp

layer_activation.o: $(SRC_PATH)/layers/layer_activation.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_activation.cpp

layer_add.o: $(SRC_PATH)/layers/layer_add.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_add.cpp

layer_average.o: $(SRC_PATH)/layers/layer_average.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_average.cpp

layer_matmul.o: $(SRC_PATH)/layers/layer_matmul.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_matmul.cpp

layer_maximum.o: $(SRC_PATH)/layers/layer_maximum.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_maximum.cpp

layer_minimum.o: $(SRC_PATH)/layers/layer_minimum.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_minimum.cpp

layer_subtract.o: $(SRC_PATH)/layers/layer_subtract.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_subtract.cpp

layer_cat.o: $(SRC_PATH)/layers/layer_cat.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_cat.cpp

layer_drop.o: $(SRC_PATH)/layers/layer_drop.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_drop.cpp

loss.o: $(SRC_PATH)/loss.cpp $(SRC_PATH)/loss.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/loss.cpp

metric.o: $(SRC_PATH)/metric.cpp $(SRC_PATH)/metric.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/metric.cpp

optim.o: $(SRC_PATH)/optim.cpp $(SRC_PATH)/optim.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optim.cpp

net.o: $(SRC_PATH)/net.cpp $(SRC_PATH)/net.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/net.cpp

eddl.o: $(SRC_PATH)/eddl.cpp $(SRC_PATH)/eddl.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/eddl.cpp

layer_reshape.o: $(SRC_PATH)/layers/layer_reshape.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer_reshape.cpp

wrapper.o: $(SRC_PATH)/eddl.cpp $(SRC_PATH)/wrapper.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/wrapper.cpp

##ADD_MORE_RULES

#---------------------------------- gpu code
tensor_cuda.o: $(SRC_PATH)/gpu/tensor_cuda.cu $(SRC_PATH)/gpu/tensor_cuda.h
		$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/gpu/tensor_cuda.cu

tensor_cuda_op.o: $(SRC_PATH)/gpu/tensor_cuda_op.cu $(SRC_PATH)/gpu/tensor_cuda_op.h
		$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/gpu/tensor_cuda_op.cu

tensor_kernels.o: $(SRC_PATH)/gpu/tensor_kernels.cu $(SRC_PATH)/gpu/tensor_kernels.h
		$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/gpu/tensor_kernels.cu

#---------------------------------- release
test_eddl: test_eddl.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) test_eddl.cpp -o test_eddl $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

test_wrapper: test_wrapper.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) test_wrapper.cpp -o test_wrapper $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

#-------------------------

install:
	cp test_eddl /usr/local/bin
clean:
	rm *.o





###########################################################################
