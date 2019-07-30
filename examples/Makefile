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
CXXSOFLAGS = -shared
ifeq ($(OPT),1)
$(info set AVX)
CXXFLAGS = -std=c++11 -march=native -mavx -O3 -pthread
endif
ifeq ($(OPT),2)
$(info set SSE)
CXXFLAGS = -std=c++11 -march=native -msse -O3 -pthread
endif
ifeq ($(OPT),3)
$(info set no opt)
CXXFLAGS = -std=c++11 -O3 -pthread
endif
ifeq ($(OPT),4)
$(info set Debug)
CXXFLAGS = -std=c++11 -g -pthread
endif
LIBFLAGS = -Wl,--no-as-needed -pthread
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

SRC_PATH=../src

# Group compiled objects (Alphabetically sorted!)
APIS_OBJ: eddl.o
CORE_OBJ: compserv.o net.o utils.o
HARDWARE_OBJ: cpu_convol.o
INITIALIZERS_OBJ: initializer_indentity.o initializer_random_uniform.o initializer_glorot_uniform.o initializer_orthogonal.o initializer_random_normal.o initializer_constant.o initializer_glorot_normal.o initializer.o
LAYERS_OBJ: layer_lstm.o layer_input.o layer_conv.o layer_mean.o layer_log10.o layer_reshape.o layer_log.o layer_mult.o layer_maxpool.o layer_embedding.o layer_globalavgpool.o layer_upsampling.o layer_add.o layer_batchnorm.o layer_div.o layer_log2.o layer_gaussian.o layer_average.o layer.o layer_globalmaxpool.o layer_reduction.o layer_sum.o layer_gauss.o layer_transposed_conv.o layer_activation.o layer_operator.o layer_concat.o layer_maximum.o layer_diff.o layer_var.o layer_drop.o layer_pool.o layer_exp.o layer_rnn.o layer_min.o layer_pow.o layer_dense.o layer_sqrt.o layer_tensor.o layer_minimum.o layer_abs.o layer_avgpool.o layer_matmul.o layer_uniform.o layer_max.o layer_subtract.o layer_generator.o layer_transpose.o
LOSSES_OBJ: loss_mean_squared_error.o loss_soft_cross_entropy.o loss.o loss_cross_entropy.o
METRICS_OBJ: metric_categorical_acc.o metric.o metric_mean_squared_error.o
OPTIMIZERS_OBJ: optim_adam.o optim.o optim_nadam.o optim_adagrad.o optim_sgd.o optim_rmsprop.o optim_adamax.o optim_adadelta.o
TENSOR_OBJ: tensor_unary.o tensor_zero.o tensor_binary.o tensor.o tensor_operators.o

# Join groups
OBJ := $(APIS_OBJ) $(CORE_OBJ) $(HARDWARE_OBJ) $(INITIALIZERS_OBJ) $(LAYERS_OBJ) $(LOSSES_OBJ) $(METRICS_OBJ) $(OPTIMIZERS_OBJ) $(TENSOR_OBJ)

ifeq ($(CORE),1)  # CPU
all: $(GPU_OBJ) $(OBJ) shared_eddl eddl_mlp eddl_dae_class eddl_predict eddl_dae eddl_conv eddl_ae eddl_test eddl_gpu eddl_load_save
endif

ifeq ($(CORE),2)  # GPU
all: $(GPU_OBJ) $(OBJ) shared_eddl eddl_mlp eddl_dae_class eddl_predict eddl_dae eddl_conv eddl_ae eddl_test eddl_gpu eddl_load_save
endif

utils.o: $(SRC_PATH)/utils.cpp $(SRC_PATH)/utils.h $(SRC_PATH)/system_info.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/utils.cpp

compserv.o: $(SRC_PATH)/compserv.cpp $(SRC_PATH)/compserv.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/compserv.cpp

net.o: $(SRC_PATH)/net.cpp $(SRC_PATH)/net.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/net.cpp

tensor_unary.o: $(SRC_PATH)/tensor/tensor_unary.cpp $(SRC_PATH)/tensor/tensor_unary.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/tensor_unary.cpp

tensor_zero.o: $(SRC_PATH)/tensor/tensor_zero.cpp $(SRC_PATH)/tensor/tensor_zero.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/tensor_zero.cpp

tensor_operators.o: $(SRC_PATH)/tensor/tensor_operators.cpp $(SRC_PATH)/tensor/tensor_operators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/tensor_operators.cpp

tensor_binary.o: $(SRC_PATH)/tensor/tensor_binary.cpp $(SRC_PATH)/tensor/tensor_binary.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/tensor_binary.cpp

tensor.o: $(SRC_PATH)/tensor/tensor.cpp $(SRC_PATH)/tensor/tensor.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/tensor.cpp

cpu_convol.o: $(SRC_PATH)/hardware/cpu/cpu_convol.cpp $(SRC_PATH)/hardware/cpu/cpu_convol.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/cpu_convol.cpp

optim_adam.o: $(SRC_PATH)/optimizers/optim_adam.cpp $(SRC_PATH)/optimizers/optim_adam.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_adam.cpp

optim_adamax.o: $(SRC_PATH)/optimizers/optim_adamax.cpp $(SRC_PATH)/optimizers/optim_adamax.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_adamax.cpp

optim_sgd.o: $(SRC_PATH)/optimizers/optim_sgd.cpp $(SRC_PATH)/optimizers/optim_sgd.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_sgd.cpp

optim_rmsprop.o: $(SRC_PATH)/optimizers/optim_rmsprop.cpp $(SRC_PATH)/optimizers/optim_rmsprop.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_rmsprop.cpp

optim.o: $(SRC_PATH)/optimizers/optim.cpp $(SRC_PATH)/optimizers/optim.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim.cpp

optim_adagrad.o: $(SRC_PATH)/optimizers/optim_adagrad.cpp $(SRC_PATH)/optimizers/optim_adagrad.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_adagrad.cpp

optim_nadam.o: $(SRC_PATH)/optimizers/optim_nadam.cpp $(SRC_PATH)/optimizers/optim_nadam.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_nadam.cpp

optim_adadelta.o: $(SRC_PATH)/optimizers/optim_adadelta.cpp $(SRC_PATH)/optimizers/optim_adadelta.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_adadelta.cpp

eddl.o: $(SRC_PATH)/apis/eddl.cpp $(SRC_PATH)/apis/eddl.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/apis/eddl.cpp

layer.o: $(SRC_PATH)/layers/layer.cpp $(SRC_PATH)/layers/layer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer.cpp

layer_maxpool.o: $(SRC_PATH)/layers/pool/layer_maxpool.cpp $(SRC_PATH)/layers/pool/layer_maxpool.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_maxpool.cpp

layer_globalmaxpool.o: $(SRC_PATH)/layers/pool/layer_globalmaxpool.cpp $(SRC_PATH)/layers/pool/layer_globalmaxpool.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_globalmaxpool.cpp

layer_pool.o: $(SRC_PATH)/layers/pool/layer_pool.cpp $(SRC_PATH)/layers/pool/layer_pool.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_pool.cpp

layer_avgpool.o: $(SRC_PATH)/layers/pool/layer_avgpool.cpp $(SRC_PATH)/layers/pool/layer_avgpool.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_avgpool.cpp

layer_globalavgpool.o: $(SRC_PATH)/layers/pool/layer_globalavgpool.cpp $(SRC_PATH)/layers/pool/layer_globalavgpool.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_globalavgpool.cpp

layer_drop.o: $(SRC_PATH)/layers/core/layer_drop.cpp $(SRC_PATH)/layers/core/layer_drop.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_drop.cpp

layer_activation.o: $(SRC_PATH)/layers/core/layer_activation.cpp $(SRC_PATH)/layers/core/layer_activation.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_activation.cpp

layer_batchnorm.o: $(SRC_PATH)/layers/core/layer_batchnorm.cpp $(SRC_PATH)/layers/core/layer_batchnorm.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_batchnorm.cpp

layer_transpose.o: $(SRC_PATH)/layers/core/layer_transpose.cpp $(SRC_PATH)/layers/core/layer_transpose.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_transpose.cpp

layer_embedding.o: $(SRC_PATH)/layers/core/layer_embedding.cpp $(SRC_PATH)/layers/core/layer_embedding.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_embedding.cpp

layer_dense.o: $(SRC_PATH)/layers/core/layer_dense.cpp $(SRC_PATH)/layers/core/layer_dense.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_dense.cpp

layer_reshape.o: $(SRC_PATH)/layers/core/layer_reshape.cpp $(SRC_PATH)/layers/core/layer_reshape.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_reshape.cpp

layer_tensor.o: $(SRC_PATH)/layers/core/layer_tensor.cpp $(SRC_PATH)/layers/core/layer_tensor.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_tensor.cpp

layer_input.o: $(SRC_PATH)/layers/core/layer_input.cpp $(SRC_PATH)/layers/core/layer_input.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_input.cpp

layer_concat.o: $(SRC_PATH)/layers/merge/layer_concat.cpp $(SRC_PATH)/layers/merge/layer_concat.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_concat.cpp

layer_matmul.o: $(SRC_PATH)/layers/merge/layer_matmul.cpp $(SRC_PATH)/layers/merge/layer_matmul.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_matmul.cpp

layer_maximum.o: $(SRC_PATH)/layers/merge/layer_maximum.cpp $(SRC_PATH)/layers/merge/layer_maximum.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_maximum.cpp

layer_subtract.o: $(SRC_PATH)/layers/merge/layer_subtract.cpp $(SRC_PATH)/layers/merge/layer_subtract.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_subtract.cpp

layer_add.o: $(SRC_PATH)/layers/merge/layer_add.cpp $(SRC_PATH)/layers/merge/layer_add.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_add.cpp

layer_average.o: $(SRC_PATH)/layers/merge/layer_average.cpp $(SRC_PATH)/layers/merge/layer_average.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_average.cpp

layer_minimum.o: $(SRC_PATH)/layers/merge/layer_minimum.cpp $(SRC_PATH)/layers/merge/layer_minimum.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_minimum.cpp

layer_div.o: $(SRC_PATH)/layers/operators/layer_div.cpp $(SRC_PATH)/layers/operators/layer_div.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_div.cpp

layer_diff.o: $(SRC_PATH)/layers/operators/layer_diff.cpp $(SRC_PATH)/layers/operators/layer_diff.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_diff.cpp

layer_pow.o: $(SRC_PATH)/layers/operators/layer_pow.cpp $(SRC_PATH)/layers/operators/layer_pow.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_pow.cpp

layer_exp.o: $(SRC_PATH)/layers/operators/layer_exp.cpp $(SRC_PATH)/layers/operators/layer_exp.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_exp.cpp

layer_sqrt.o: $(SRC_PATH)/layers/operators/layer_sqrt.cpp $(SRC_PATH)/layers/operators/layer_sqrt.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_sqrt.cpp

layer_log.o: $(SRC_PATH)/layers/operators/layer_log.cpp $(SRC_PATH)/layers/operators/layer_log.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_log.cpp

layer_log10.o: $(SRC_PATH)/layers/operators/layer_log10.cpp $(SRC_PATH)/layers/operators/layer_log10.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_log10.cpp

layer_operator.o: $(SRC_PATH)/layers/operators/layer_operator.cpp $(SRC_PATH)/layers/operators/layer_operator.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_operator.cpp

layer_abs.o: $(SRC_PATH)/layers/operators/layer_abs.cpp $(SRC_PATH)/layers/operators/layer_abs.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_abs.cpp

layer_sum.o: $(SRC_PATH)/layers/operators/layer_sum.cpp $(SRC_PATH)/layers/operators/layer_sum.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_sum.cpp

layer_mult.o: $(SRC_PATH)/layers/operators/layer_mult.cpp $(SRC_PATH)/layers/operators/layer_mult.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_mult.cpp

layer_log2.o: $(SRC_PATH)/layers/operators/layer_log2.cpp $(SRC_PATH)/layers/operators/layer_log2.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_log2.cpp

layer_generator.o: $(SRC_PATH)/layers/generators/layer_generator.cpp $(SRC_PATH)/layers/generators/layer_generator.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/generators/layer_generator.cpp

layer_gauss.o: $(SRC_PATH)/layers/generators/layer_gauss.cpp $(SRC_PATH)/layers/generators/layer_gauss.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/generators/layer_gauss.cpp

layer_uniform.o: $(SRC_PATH)/layers/generators/layer_uniform.cpp $(SRC_PATH)/layers/generators/layer_uniform.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/generators/layer_uniform.cpp

layer_gaussian.o: $(SRC_PATH)/layers/noise/layer_gaussian.cpp $(SRC_PATH)/layers/noise/layer_gaussian.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/noise/layer_gaussian.cpp

layer_upsampling.o: $(SRC_PATH)/layers/conv/layer_upsampling.cpp $(SRC_PATH)/layers/conv/layer_upsampling.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/conv/layer_upsampling.cpp

layer_conv.o: $(SRC_PATH)/layers/conv/layer_conv.cpp $(SRC_PATH)/layers/conv/layer_conv.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/conv/layer_conv.cpp

layer_transposed_conv.o: $(SRC_PATH)/layers/conv/layer_transposed_conv.cpp $(SRC_PATH)/layers/conv/layer_transposed_conv.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/conv/layer_transposed_conv.cpp

layer_lstm.o: $(SRC_PATH)/layers/recurrent/layer_lstm.cpp $(SRC_PATH)/layers/recurrent/layer_lstm.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/recurrent/layer_lstm.cpp

layer_rnn.o: $(SRC_PATH)/layers/recurrent/layer_rnn.cpp $(SRC_PATH)/layers/recurrent/layer_rnn.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/recurrent/layer_rnn.cpp

layer_reduction.o: $(SRC_PATH)/layers/reductions/layer_reduction.cpp $(SRC_PATH)/layers/reductions/layer_reduction.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/reductions/layer_reduction.cpp

layer_var.o: $(SRC_PATH)/layers/reductions/layer_var.cpp $(SRC_PATH)/layers/reductions/layer_var.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/reductions/layer_var.cpp

layer_max.o: $(SRC_PATH)/layers/reductions/layer_max.cpp $(SRC_PATH)/layers/reductions/layer_max.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/reductions/layer_max.cpp

layer_min.o: $(SRC_PATH)/layers/reductions/layer_min.cpp $(SRC_PATH)/layers/reductions/layer_min.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/reductions/layer_min.cpp

layer_sum.o: $(SRC_PATH)/layers/reductions/layer_sum.cpp $(SRC_PATH)/layers/reductions/layer_sum.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/reductions/layer_sum.cpp

layer_mean.o: $(SRC_PATH)/layers/reductions/layer_mean.cpp $(SRC_PATH)/layers/reductions/layer_mean.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/reductions/layer_mean.cpp

loss_mean_squared_error.o: $(SRC_PATH)/losses/loss_mean_squared_error.cpp $(SRC_PATH)/losses/loss_mean_squared_error.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/losses/loss_mean_squared_error.cpp

loss_soft_cross_entropy.o: $(SRC_PATH)/losses/loss_soft_cross_entropy.cpp $(SRC_PATH)/losses/loss_soft_cross_entropy.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/losses/loss_soft_cross_entropy.cpp

loss.o: $(SRC_PATH)/losses/loss.cpp $(SRC_PATH)/losses/loss.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/losses/loss.cpp

loss_cross_entropy.o: $(SRC_PATH)/losses/loss_cross_entropy.cpp $(SRC_PATH)/losses/loss_cross_entropy.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/losses/loss_cross_entropy.cpp

metric_categorical_acc.o: $(SRC_PATH)/metrics/metric_categorical_acc.cpp $(SRC_PATH)/metrics/metric_categorical_acc.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/metrics/metric_categorical_acc.cpp

metric.o: $(SRC_PATH)/metrics/metric.cpp $(SRC_PATH)/metrics/metric.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/metrics/metric.cpp

metric_mean_squared_error.o: $(SRC_PATH)/metrics/metric_mean_squared_error.cpp $(SRC_PATH)/metrics/metric_mean_squared_error.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/metrics/metric_mean_squared_error.cpp

initializer_random_normal.o: $(SRC_PATH)/initializers/initializer_random_normal.cpp $(SRC_PATH)/initializers/initializer_random_normal.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer_random_normal.cpp

initializer_indentity.o: $(SRC_PATH)/initializers/initializer_indentity.cpp $(SRC_PATH)/initializers/initializer_indentity.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer_indentity.cpp

initializer_glorot_normal.o: $(SRC_PATH)/initializers/initializer_glorot_normal.cpp $(SRC_PATH)/initializers/initializer_glorot_normal.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer_glorot_normal.cpp

initializer_glorot_uniform.o: $(SRC_PATH)/initializers/initializer_glorot_uniform.cpp $(SRC_PATH)/initializers/initializer_glorot_uniform.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer_glorot_uniform.cpp

initializer_orthogonal.o: $(SRC_PATH)/initializers/initializer_orthogonal.cpp $(SRC_PATH)/initializers/initializer_orthogonal.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer_orthogonal.cpp

initializer_random_uniform.o: $(SRC_PATH)/initializers/initializer_random_uniform.cpp $(SRC_PATH)/initializers/initializer_random_uniform.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer_random_uniform.cpp

initializer_constant.o: $(SRC_PATH)/initializers/initializer_constant.cpp $(SRC_PATH)/initializers/initializer_constant.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer_constant.cpp

initializer.o: $(SRC_PATH)/initializers/initializer.cpp $(SRC_PATH)/initializers/initializer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer.cpp


#---------------------------------- GPU code
tensor_cuda.o: $(SRC_PATH)/hardware/gpu/tensor_cuda.cu $(SRC_PATH)/hardware/gpu/tensor_cuda.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/tensor_cuda.cu

tensor_kernels.o: $(SRC_PATH)/hardware/gpu/tensor_kernels.cu $(SRC_PATH)/hardware/gpu/tensor_kernels.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/tensor_kernels.cu

tensor_cuda_op.o: $(SRC_PATH)/hardware/gpu/tensor_cuda_op.cu $(SRC_PATH)/hardware/gpu/tensor_cuda_op.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/tensor_cuda_op.cu


#---------------------------------- release
shared_eddl: $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(CXXSOFLAGS) $(GPU_CXXFLAGS) -o eddl.so $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

eddl_mlp: eddl_mlp.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) eddl_mlp.cpp -o eddl_mlp $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

eddl_dae_class: eddl_dae_class.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) eddl_dae_class.cpp -o eddl_dae_class $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

eddl_predict: eddl_predict.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) eddl_predict.cpp -o eddl_predict $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

eddl_dae: eddl_dae.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) eddl_dae.cpp -o eddl_dae $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

eddl_conv: eddl_conv.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) eddl_conv.cpp -o eddl_conv $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

eddl_ae: eddl_ae.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) eddl_ae.cpp -o eddl_ae $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

eddl_test: eddl_test.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) eddl_test.cpp -o eddl_test $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

eddl_gpu: eddl_gpu.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) eddl_gpu.cpp -o eddl_gpu $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

eddl_load_save: eddl_load_save.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) eddl_load_save.cpp -o eddl_load_save $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)


#---------------------------------- Extra options
install:
	cp test_eddl /usr/local/bin
clean:
	rm *.o