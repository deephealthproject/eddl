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
CXXSOFLAGS = -shared
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

# Create position independent code for the shared library
CXXFLAGS := $(CXXFLAGS) -fPIC


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

# Group compiled objects (Alphabetically sorted!)
APIS_OBJ = eddl.o wrapper.o
HARDWARE_OBJ = cpu_convol.o
CONV_LAYERS_OBJ = layer_conv.o layer_transposed_conv.o layer_upsampling.o
CORE_LAYERS_OBJ = layer.o layer_activation.o layer_batchnorm.o layer_dense.o layer_drop.o layer_embedding.o layer_input.o layer_reshape.o layer_tensor.o layer_transpose.o
MERGE_LAYERS_OBJ = layer_add.o layer_average.o layer_concat.o layer_matmul.o layer_maximum.o layer_minimum.o layer_subtract.o
NOISE_LAYERS_OBJ = layer_gaussian.o
OPERATORS_LAYERS_OBJ = layer_operator.o layer_abs.o layer_diff.o layer_div.o layer_exp.o layer_log.o layer_mean.o layer_mult.o layer_pow.o layer_sqrt.o layer_sum.o layer_var.o
POOL_LAYERS_OBJ = layer_globalmaxpool.o layer_avgpool.o layer_globalavgpool.o layer_maxpool.o layer_pool.o
RECURRENT_LAYERS_OBJ = layer_rnn.o layer_lstm.o
LOSSES_OBJ = loss.o loss_cross_entropy.o loss_mean_squared_error.o loss_soft_cross_entropy.o
METRICS_OBJ = metric.o metric_categorical_acc.o metric_mean_squared_error.o
OPTIMIZERS_OBJ = optim.o optim_adadelta.o optim_adagrad.o optim_adam.o optim_adamax.o optim_nadam.o optim_rmsprop.o optim_sgd.o
CORE_OBJ = compserv.o initializer.o net.o tensor.o tensor_op.o utils.o

# Join groups
OBJ := $(APIS_OBJ) $(HARDWARE_OBJ) $(CONV_LAYERS_OBJ) $(CORE_LAYERS_OBJ) $(MERGE_LAYERS_OBJ) $(NOISE_LAYERS_OBJ) $(OPERATORS_LAYERS_OBJ) $(POOL_LAYERS_OBJ) $(RECURRENT_LAYERS_OBJ) $(LOSSES_OBJ) $(METRICS_OBJ) $(OPTIMIZERS_OBJ) $(CORE_OBJ)

ifeq ($(CORE),1)  # CPU
all: $(GPU_OBJ) $(OBJ) shared_eddl test_eddl test_wrapper
endif

ifeq ($(CORE),2)  # GPU
all: $(GPU_OBJ) $(OBJ) shared_eddl test_eddl test_wrapper
endif

#---------------------------------- apis
eddl.o: $(SRC_PATH)/apis/eddl.cpp $(SRC_PATH)/apis/eddl.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/apis/eddl.cpp
wrapper.o: $(SRC_PATH)/apis/eddl.cpp $(SRC_PATH)/apis/wrapper.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/apis/wrapper.cpp

#---------------------------------- hardware
cpu_convol.o: $(SRC_PATH)/hardware/cpu/cpu_convol.cpp $(SRC_PATH)/hardware/cpu/cpu_convol.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/cpu_convol.cpp

#---------------------------------- core layers
layer.o: $(SRC_PATH)/layers/layer.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer.cpp
layer_activation.o: $(SRC_PATH)/layers/core/layer_activation.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_activation.cpp
layer_batchnorm.o: $(SRC_PATH)/layers/core/layer_batchnorm.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_batchnorm.cpp
layer_dense.o: $(SRC_PATH)/layers/core/layer_dense.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_dense.cpp
layer_drop.o: $(SRC_PATH)/layers/core/layer_drop.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_drop.cpp
layer_embedding.o: $(SRC_PATH)/layers/core/layer_embedding.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_embedding.cpp
layer_input.o: $(SRC_PATH)/layers/core/layer_input.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_input.cpp
layer_tensor.o: $(SRC_PATH)/layers/core/layer_tensor.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_tensor.cpp

#---------------------------------- conv layers
layer_conv.o: $(SRC_PATH)/layers/conv/layer_conv.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/conv/layer_conv.cpp
layer_transposed_conv.o: $(SRC_PATH)/layers/conv/layer_transposed_conv.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/conv/layer_transposed_conv.cpp
layer_upsampling.o: $(SRC_PATH)/layers/conv/layer_conv.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/conv/layer_upsampling.cpp

#---------------------------------- merge layers
layer_add.o: $(SRC_PATH)/layers/merge/layer_add.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_add.cpp
layer_average.o: $(SRC_PATH)/layers/merge/layer_average.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_average.cpp
layer_concat.o: $(SRC_PATH)/layers/merge/layer_concat.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_concat.cpp
layer_matmul.o: $(SRC_PATH)/layers/merge/layer_matmul.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_matmul.cpp
layer_maximum.o: $(SRC_PATH)/layers/merge/layer_maximum.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_maximum.cpp
layer_minimum.o: $(SRC_PATH)/layers/merge/layer_minimum.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_minimum.cpp
layer_subtract.o: $(SRC_PATH)/layers/merge/layer_subtract.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_subtract.cpp

#---------------------------------- noise layers
layer_gaussian.o: $(SRC_PATH)/layers/noise/layer_gaussian.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/noise/layer_gaussian.cpp

#---------------------------------- operator layers
layer_operator.o: $(SRC_PATH)/layers/operators/layer_operator.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_operator.cpp
layer_abs.o: $(SRC_PATH)/layers/operators/layer_abs.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_abs.cpp
layer_diff.o: $(SRC_PATH)/layers/operators/layer_diff.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_diff.cpp
layer_div.o: $(SRC_PATH)/layers/operators/layer_div.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_div.cpp
layer_exp.o: $(SRC_PATH)/layers/operators/layer_exp.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_exp.cpp
layer_log.o: $(SRC_PATH)/layers/operators/layer_log.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_log.cpp
layer_mean.o: $(SRC_PATH)/layers/operators/layer_mean.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_mean.cpp
layer_mult.o: $(SRC_PATH)/layers/operators/layer_mult.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_mult.cpp
layer_pow.o: $(SRC_PATH)/layers/operators/layer_pow.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_pow.cpp
layer_reshape.o: $(SRC_PATH)/layers/operators/layer_reshape.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_reshape.cpp
layer_sqrt.o: $(SRC_PATH)/layers/operators/layer_sqrt.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_sqrt.cpp
layer_sum.o: $(SRC_PATH)/layers/operators/layer_sum.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_sum.cpp
layer_transpose.o: $(SRC_PATH)/layers/operators/layer_transpose.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_transpose.cpp
layer_var.o: $(SRC_PATH)/layers/operators/layer_var.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_var.cpp

#---------------------------------- pool layers
layer_avgpool.o: $(SRC_PATH)/layers/pool/layer_avgpool.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_avgpool.cpp
layer_globalavgpool.o: $(SRC_PATH)/layers/pool/layer_globalavgpool.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_globalavgpool.cpp
layer_globalmaxpool.o: $(SRC_PATH)/layers/pool/layer_globalmaxpool.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_globalmaxpool.cpp
layer_maxpool.o: $(SRC_PATH)/layers/pool/layer_maxpool.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_maxpool.cpp
layer_pool.o: $(SRC_PATH)/layers/pool/layer_pool.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_pool.cpp

#---------------------------------- recurrent layers
layer_rnn.o: $(SRC_PATH)/layers/recurrent/layer_rnn.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/recurrent/layer_rnn.cpp
layer_lstm.o: $(SRC_PATH)/layers/recurrent/layer_lstm.cpp $(SRC_PATH)/layers/layer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/recurrent/layer_lstm.cpp

#---------------------------------- losses
loss.o: $(SRC_PATH)/losses/loss.cpp $(SRC_PATH)/losses/loss.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/losses/loss.cpp
loss_cross_entropy.o: $(SRC_PATH)/losses/loss_cross_entropy.cpp $(SRC_PATH)/losses/loss.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/losses/loss_cross_entropy.cpp
loss_mean_squared_error.o: $(SRC_PATH)/losses/loss_mean_squared_error.cpp $(SRC_PATH)/losses/loss.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/losses/loss_mean_squared_error.cpp
loss_soft_cross_entropy.o: $(SRC_PATH)/losses/loss_soft_cross_entropy.cpp $(SRC_PATH)/losses/loss.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/losses/loss_soft_cross_entropy.cpp

#---------------------------------- metrics
metric.o: $(SRC_PATH)/metrics/metric.cpp $(SRC_PATH)/metrics/metric.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/metrics/metric.cpp
metric_categorical_acc.o: $(SRC_PATH)/metrics/metric_categorical_acc.cpp $(SRC_PATH)/metrics/metric.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/metrics/metric_categorical_acc.cpp
metric_mean_squared_error.o: $(SRC_PATH)/metrics/metric_mean_squared_error.cpp $(SRC_PATH)/metrics/metric.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/metrics/metric_mean_squared_error.cpp

#---------------------------------- optimizers
optim.o: $(SRC_PATH)/optimizers/optim.cpp $(SRC_PATH)/optimizers/optim.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim.cpp
optim_adadelta.o: $(SRC_PATH)/optimizers/optim_adadelta.cpp $(SRC_PATH)/optimizers/optim.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_adadelta.cpp
optim_adagrad.o: $(SRC_PATH)/optimizers/optim_adagrad.cpp $(SRC_PATH)/optimizers/optim.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_adagrad.cpp
optim_adam.o: $(SRC_PATH)/optimizers/optim_adam.cpp $(SRC_PATH)/optimizers/optim.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_adam.cpp
optim_adamax.o: $(SRC_PATH)/optimizers/optim_adamax.cpp $(SRC_PATH)/optimizers/optim.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_adamax.cpp
optim_nadam.o: $(SRC_PATH)/optimizers/optim_nadam.cpp $(SRC_PATH)/optimizers/optim.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_nadam.cpp
optim_rmsprop.o: $(SRC_PATH)/optimizers/optim_rmsprop.cpp $(SRC_PATH)/optimizers/optim.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_rmsprop.cpp
optim_sgd.o: $(SRC_PATH)/optimizers/optim_sgd.cpp $(SRC_PATH)/optimizers/optim.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_sgd.cpp

#---------------------------------- core classes
compserv.o: $(SRC_PATH)/compserv.cpp $(SRC_PATH)/compserv.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/compserv.cpp
initializer.o: $(SRC_PATH)/initializer.cpp $(SRC_PATH)/initializer.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializer.cpp
net.o: $(SRC_PATH)/net.cpp $(SRC_PATH)/net.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/net.cpp
tensor.o: $(SRC_PATH)/tensor.cpp $(SRC_PATH)/tensor.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor.cpp
tensor_op.o: $(SRC_PATH)/tensor_op.cpp $(SRC_PATH)/tensor.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor_op.cpp
utils.o: $(SRC_PATH)/utils.cpp $(SRC_PATH)/utils.h
		$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/utils.cpp

#---------------------------------- GPU code
tensor_cuda.o: $(SRC_PATH)/hardware/gpu/tensor_cuda.cu $(SRC_PATH)/hardware/gpu/tensor_cuda.h
		$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/tensor_cuda.cu

tensor_cuda_op.o: $(SRC_PATH)/hardware/gpu/tensor_cuda_op.cu $(SRC_PATH)/hardware/gpu/tensor_cuda_op.h
		$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/tensor_cuda_op.cu

tensor_kernels.o: $(SRC_PATH)/hardware/gpu/tensor_kernels.cu $(SRC_PATH)/hardware/gpu/tensor_kernels.h
		$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/tensor_kernels.cu

#---------------------------------- release
shared_eddl: $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(CXXSOFLAGS) $(GPU_CXXFLAGS) -o eddl.so $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

test_eddl: test_eddl.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) test_eddl.cpp -o test_eddl $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

test_wrapper: test_wrapper.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) test_wrapper.cpp -o test_wrapper $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

#---------------------------------- Extra options
install:
	cp test_eddl /usr/local/bin
clean:
	rm *.o
