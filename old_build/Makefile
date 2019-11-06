ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),install)

# Default paths
SRC_PATH=../src
EXAMPLES_NN_PATH=../examples/NN
EXAMPLES_T_PATH=../examples/Tensor

TESTS_PATH=../tests
BUILD_PATH=.
LIBS_PATH=../third_party/eigen/


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
CXXFLAGS = -std=c++11 -mavx -O3 -pthread
endif
ifeq ($(OPT),2)
$(info set SSE)
CXXFLAGS = -std=c++11 -msse -O3 -pthread
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

#CXX = g++ #clang
CXX = /usr/local/bin/g++-9 # OpenMP
CXXSOFLAGS = -shared
ifeq ($(OPT),1)
$(info set AVX)
CXXFLAGS =-std=c++11 -mavx -O3
endif
ifeq ($(OPT),2)
$(info set SSE)
CXXFLAGS =-std=c++11 -msse -O3
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
#CXXFLAGS := $(CXXFLAGS) -fPIC
CXXFLAGS := $(CXXFLAGS) -fopenmp -fPIC


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
GPU_OBJ = gpu_tensor.o gpu_activations_kernels.o gpu_conv_kernels.o gpu_tensor_nn_kernels.o gpu_losses_kernels.o gpu_metrics_kernels.o gpu_pool_kernels.o gpu_core_kernels.o gpu_create_kernels.o gpu_da_kernels.o gpu_generator_kernels.o gpu_math_kernels.o gpu_reduction_kernels.o gpu_comparison.o gpu_core.o gpu_create.o gpu_da.o gpu_generator.o gpu_math.o gpu_reduction.o gpu_activations.o gpu_conv.o gpu_tensor_nn.o gpu_losses.o gpu_metrics.o gpu_pool.o

endif
endif
endif


# EXAMPLES
ifndef EXAMPLES
EXAMPLES = test_main example_mlp example_conv example_ae example_dae example_dae_class example_tensor example_load_save example_train_batch
endif
$(info set EXAMPLES $(EXAMPLES))


# Group compiled objects (Alphabetically sorted!)
APIS_OBJ = eddl.o eddlT.o
CONSTRAINTS_OBJ = constraint_max_norm.o constraint_min_max_norm.o constraint_non_neg.o constraint_unit_norm.o constraint.o
CORE_OBJ = compserv.o net.o utils.o random.o
DESCRIPTORS_OBJ = descriptor_conv.o descriptor_pool.o descriptor_reduce.o
HARDWARE_CPU_OBJ = cpu_comparison.o cpu_core.o cpu_create.o cpu_da.o cpu_generator.o cpu_math.o cpu_reduction.o cpu_activations.o cpu_conv.o cpu_losses.o cpu_metrics.o cpu_pool.o cpu_tensor_nn.o
INITIALIZERS_OBJ = initializer_indentity.o initializer_random_uniform.o initializer_glorot_uniform.o initializer_orthogonal.o initializer_random_normal.o initializer_constant.o initializer_glorot_normal.o initializer.o
LAYERS_OBJ = layer_crop.o layer_crop_random.o layer_crop_scale.o layer_crop_scale_random.o layer_cutout.o layer_cutout_random.o layer_flip.o layer_flip_random.o layer_rotate.o layer_rotate_random.o layer_scale.o layer_scale_random.o layer_shift.o layer_shift_random.o layer_lstm.o layer_input.o layer_conv.o layer_mean.o layer_log10.o layer_reshape.o layer_log.o layer_mult.o layer_maxpool.o layer_embedding.o layer_globalavgpool.o layer_upsampling.o layer_add.o layer_normmax.o layer_normminmax.o layer_norm.o layer_batchnorm.o layer_div.o layer_log2.o layer_gaussian.o layer_average.o layer.o layer_globalmaxpool.o layer_reduction.o layer_sum.o layer_sum_op.o layer_gauss.o layer_transposed_conv.o layer_activation.o layer_operator.o layer_concat.o layer_maximum.o layer_diff.o layer_var.o layer_drop.o layer_pool.o layer_exp.o layer_rnn.o layer_min.o layer_pow.o layer_dense.o layer_sqrt.o layer_tensor.o layer_minimum.o layer_abs.o layer_avgpool.o layer_matmul.o layer_uniform.o layer_max.o layer_subtract.o layer_generator.o layer_transpose.o
LOSSES_OBJ = loss_mean_squared_error.o loss_soft_cross_entropy.o loss.o loss_cross_entropy.o
METRICS_OBJ = metric.o metric_categorical_acc.o metric_mean_absolute_error.o metric_mean_relative_error.o metric_mean_squared_error.o
OPTIMIZERS_OBJ = optim_adam.o optim.o optim_nadam.o optim_adagrad.o optim_sgd.o optim_rmsprop.o optim_adamax.o optim_adadelta.o
REGULARIZERS_OBJ = regularizer_l1.o regularizer_l2.o regularizer_l1_l2.o regularizer.o
TENSOR_OBJ = tensor.o tensor_comparison.o tensor_core.o tensor_create.o tensor_da.o tensor_generator.o tensor_math.o tensor_reduction.o tensor_serialization.o tensor_activations.o tensor_conv.o tensor_core_nn.o tensor_losses.o tensor_metrics.o tensor_pool.o
TESTS_OBJ = aux_tests.o

# Join groups
OBJ := $(APIS_OBJ) $(CONSTRAINTS_OBJ) $(CORE_OBJ) $(DA_OBJ) $(DESCRIPTORS_OBJ) $(HARDWARE_CPU_OBJ) $(INITIALIZERS_OBJ) $(LAYERS_OBJ) $(LOSSES_OBJ) $(METRICS_OBJ) $(OPTIMIZERS_OBJ) $(REGULARIZERS_OBJ) $(TENSOR_OBJ) $(TESTS_OBJ)

ifeq ($(CORE),1)  # CPU
all: $(GPU_OBJ) $(OBJ) $(EXAMPLES)
endif

ifeq ($(CORE),2)  # GPU
all: $(GPU_OBJ) $(OBJ) $(EXAMPLES)
endif

utils.o: $(SRC_PATH)/utils.cpp $(SRC_PATH)/utils.h $(SRC_PATH)/system_info.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/utils.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

constraint.o: $(SRC_PATH)/layers/constraints/constraint.cpp $(SRC_PATH)/layers/constraints/constraint.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/constraints/constraint.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

constraint_max_norm.o: $(SRC_PATH)/layers/constraints/constraint_max_norm.cpp $(SRC_PATH)/layers/constraints/constraint.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/constraints/constraint_max_norm.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

constraint_min_max_norm.o: $(SRC_PATH)/layers/constraints/constraint_min_max_norm.cpp $(SRC_PATH)/layers/constraints/constraint.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/constraints/constraint_min_max_norm.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

constraint_non_neg.o: $(SRC_PATH)/layers/constraints/constraint_non_neg.cpp $(SRC_PATH)/layers/constraints/constraint.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/constraints/constraint_non_neg.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

constraint_unit_norm.o: $(SRC_PATH)/layers/constraints/constraint_unit_norm.cpp $(SRC_PATH)/layers/constraints/constraint.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/constraints/constraint_unit_norm.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

random.o: $(SRC_PATH)/random.cpp $(SRC_PATH)/random.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/random.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

compserv.o: $(SRC_PATH)/compserv.cpp $(SRC_PATH)/compserv.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/compserv.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

net.o: $(SRC_PATH)/net.cpp $(SRC_PATH)/net.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/net.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

descriptor_conv.o: $(SRC_PATH)/descriptors/descriptor_conv.cpp $(SRC_PATH)/descriptors/descriptors.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/descriptors/descriptor_conv.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

descriptor_pool.o: $(SRC_PATH)/descriptors/descriptor_pool.cpp $(SRC_PATH)/descriptors/descriptors.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/descriptors/descriptor_pool.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

descriptor_reduce.o: $(SRC_PATH)/descriptors/descriptor_reduce.cpp $(SRC_PATH)/descriptors/descriptors.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/descriptors/descriptor_reduce.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor.o: $(SRC_PATH)/tensor/tensor.cpp $(SRC_PATH)/tensor/tensor.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/tensor.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor_comparison.o: $(SRC_PATH)/tensor/tensor_comparison.cpp $(SRC_PATH)/tensor/tensor.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/tensor_comparison.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor_core.o: $(SRC_PATH)/tensor/tensor_core.cpp $(SRC_PATH)/tensor/tensor.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/tensor_core.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor_create.o: $(SRC_PATH)/tensor/tensor_create.cpp $(SRC_PATH)/tensor/tensor.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/tensor_create.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor_da.o: $(SRC_PATH)/tensor/tensor_da.cpp $(SRC_PATH)/tensor/tensor.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/tensor_da.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor_generator.o: $(SRC_PATH)/tensor/tensor_generator.cpp $(SRC_PATH)/tensor/tensor.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/tensor_generator.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor_math.o: $(SRC_PATH)/tensor/tensor_math.cpp $(SRC_PATH)/tensor/tensor.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/tensor_math.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor_reduction.o: $(SRC_PATH)/tensor/tensor_reduction.cpp $(SRC_PATH)/tensor/tensor.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/tensor_reduction.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor_serialization.o: $(SRC_PATH)/tensor/tensor_serialization.cpp $(SRC_PATH)/tensor/tensor.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/tensor_serialization.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor_activations.o: $(SRC_PATH)/tensor/nn/tensor_activations.cpp $(SRC_PATH)/tensor/nn/tensor_nn.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/nn/tensor_activations.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor_conv.o: $(SRC_PATH)/tensor/nn/tensor_conv.cpp $(SRC_PATH)/tensor/nn/tensor_nn.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/nn/tensor_conv.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor_core_nn.o: $(SRC_PATH)/tensor/nn/tensor_core_nn.cpp $(SRC_PATH)/tensor/nn/tensor_nn.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/nn/tensor_core_nn.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor_losses.o: $(SRC_PATH)/tensor/nn/tensor_losses.cpp $(SRC_PATH)/tensor/nn/tensor_nn.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/nn/tensor_losses.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor_metrics.o: $(SRC_PATH)/tensor/nn/tensor_metrics.cpp $(SRC_PATH)/tensor/nn/tensor_nn.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/nn/tensor_metrics.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

tensor_pool.o: $(SRC_PATH)/tensor/nn/tensor_pool.cpp $(SRC_PATH)/tensor/nn/tensor_nn.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/tensor/nn/tensor_pool.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

cpu_activations.o: $(SRC_PATH)/hardware/cpu/nn/cpu_activations.cpp $(SRC_PATH)/hardware/cpu/nn/cpu_nn.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/nn/cpu_activations.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

cpu_conv.o: $(SRC_PATH)/hardware/cpu/nn/cpu_conv.cpp $(SRC_PATH)/hardware/cpu/nn/cpu_nn.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/nn/cpu_conv.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

cpu_losses.o: $(SRC_PATH)/hardware/cpu/nn/cpu_losses.cpp $(SRC_PATH)/hardware/cpu/nn/cpu_nn.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/nn/cpu_losses.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

cpu_metrics.o: $(SRC_PATH)/hardware/cpu/nn/cpu_metrics.cpp $(SRC_PATH)/hardware/cpu/nn/cpu_nn.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/nn/cpu_metrics.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

cpu_pool.o: $(SRC_PATH)/hardware/cpu/nn/cpu_pool.cpp $(SRC_PATH)/hardware/cpu/nn/cpu_nn.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/nn/cpu_pool.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

cpu_tensor_nn.o: $(SRC_PATH)/hardware/cpu/nn/cpu_tensor_nn.cpp $(SRC_PATH)/hardware/cpu/nn/cpu_nn.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/nn/cpu_tensor_nn.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

cpu_comparison.o: $(SRC_PATH)/hardware/cpu/cpu_comparison.cpp $(SRC_PATH)/hardware/cpu/cpu_hw.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/cpu_comparison.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

cpu_core.o: $(SRC_PATH)/hardware/cpu/cpu_core.cpp $(SRC_PATH)/hardware/cpu/cpu_hw.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/cpu_core.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

cpu_create.o: $(SRC_PATH)/hardware/cpu/cpu_create.cpp $(SRC_PATH)/hardware/cpu/cpu_hw.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/cpu_create.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

cpu_da.o: $(SRC_PATH)/hardware/cpu/cpu_da.cpp $(SRC_PATH)/hardware/cpu/cpu_hw.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/cpu_da.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

cpu_generator.o: $(SRC_PATH)/hardware/cpu/cpu_generator.cpp $(SRC_PATH)/hardware/cpu/cpu_hw.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/cpu_generator.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

cpu_math.o: $(SRC_PATH)/hardware/cpu/cpu_math.cpp $(SRC_PATH)/hardware/cpu/cpu_hw.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/cpu_math.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

cpu_reduction.o: $(SRC_PATH)/hardware/cpu/cpu_reduction.cpp $(SRC_PATH)/hardware/cpu/cpu_hw.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/hardware/cpu/cpu_reduction.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

optim_adam.o: $(SRC_PATH)/optimizers/optim_adam.cpp $(SRC_PATH)/optimizers/optim.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_adam.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

optim_adamax.o: $(SRC_PATH)/optimizers/optim_adamax.cpp $(SRC_PATH)/optimizers/optim.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_adamax.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

optim_sgd.o: $(SRC_PATH)/optimizers/optim_sgd.cpp $(SRC_PATH)/optimizers/optim.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_sgd.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

optim_rmsprop.o: $(SRC_PATH)/optimizers/optim_rmsprop.cpp $(SRC_PATH)/optimizers/optim.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_rmsprop.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

optim.o: $(SRC_PATH)/optimizers/optim.cpp $(SRC_PATH)/optimizers/optim.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

optim_adagrad.o: $(SRC_PATH)/optimizers/optim_adagrad.cpp $(SRC_PATH)/optimizers/optim.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_adagrad.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

optim_nadam.o: $(SRC_PATH)/optimizers/optim_nadam.cpp $(SRC_PATH)/optimizers/optim.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_nadam.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

optim_adadelta.o: $(SRC_PATH)/optimizers/optim_adadelta.cpp $(SRC_PATH)/optimizers/optim.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/optimizers/optim_adadelta.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

eddl.o: $(SRC_PATH)/apis/eddl.cpp $(SRC_PATH)/apis/eddl.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/apis/eddl.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

eddlT.o: $(SRC_PATH)/apis/eddlT.cpp $(SRC_PATH)/apis/eddlT.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/apis/eddlT.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer.o: $(SRC_PATH)/layers/layer.cpp $(SRC_PATH)/layers/layer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/layer.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_maxpool.o: $(SRC_PATH)/layers/pool/layer_maxpool.cpp $(SRC_PATH)/layers/pool/layer_pool.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_maxpool.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_globalmaxpool.o: $(SRC_PATH)/layers/pool/layer_globalmaxpool.cpp $(SRC_PATH)/layers/pool/layer_pool.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_globalmaxpool.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_pool.o: $(SRC_PATH)/layers/pool/layer_pool.cpp $(SRC_PATH)/layers/pool/layer_pool.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_pool.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_avgpool.o: $(SRC_PATH)/layers/pool/layer_avgpool.cpp $(SRC_PATH)/layers/pool/layer_pool.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_avgpool.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_globalavgpool.o: $(SRC_PATH)/layers/pool/layer_globalavgpool.cpp $(SRC_PATH)/layers/pool/layer_pool.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/pool/layer_globalavgpool.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_drop.o: $(SRC_PATH)/layers/core/layer_drop.cpp $(SRC_PATH)/layers/core/layer_core.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_drop.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_activation.o: $(SRC_PATH)/layers/core/layer_activation.cpp $(SRC_PATH)/layers/core/layer_core.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_activation.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_batchnorm.o: $(SRC_PATH)/layers/normalization/layer_batchnorm.cpp $(SRC_PATH)/layers/normalization/layer_normalization.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/normalization/layer_batchnorm.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_norm.o: $(SRC_PATH)/layers/normalization/layer_norm.cpp $(SRC_PATH)/layers/normalization/layer_normalization.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/normalization/layer_norm.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_normmax.o: $(SRC_PATH)/layers/normalization/layer_normmax.cpp $(SRC_PATH)/layers/normalization/layer_normalization.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/normalization/layer_normmax.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_normminmax.o: $(SRC_PATH)/layers/normalization/layer_normminmax.cpp $(SRC_PATH)/layers/normalization/layer_normalization.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/normalization/layer_normminmax.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@


layer_transpose.o: $(SRC_PATH)/layers/core/layer_transpose.cpp $(SRC_PATH)/layers/core/layer_core.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_transpose.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_embedding.o: $(SRC_PATH)/layers/core/layer_embedding.cpp $(SRC_PATH)/layers/core/layer_core.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_embedding.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_dense.o: $(SRC_PATH)/layers/core/layer_dense.cpp $(SRC_PATH)/layers/core/layer_core.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_dense.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_reshape.o: $(SRC_PATH)/layers/core/layer_reshape.cpp $(SRC_PATH)/layers/core/layer_core.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_reshape.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_tensor.o: $(SRC_PATH)/layers/core/layer_tensor.cpp $(SRC_PATH)/layers/core/layer_core.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_tensor.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_input.o: $(SRC_PATH)/layers/core/layer_input.cpp $(SRC_PATH)/layers/core/layer_core.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/core/layer_input.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_crop.o: $(SRC_PATH)/layers/da/layer_crop.cpp $(SRC_PATH)/layers/da/layer_da.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/da/layer_crop.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_crop_random.o: $(SRC_PATH)/layers/da/layer_crop_random.cpp $(SRC_PATH)/layers/da/layer_da.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/da/layer_crop_random.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_crop_scale.o: $(SRC_PATH)/layers/da/layer_crop_scale.cpp $(SRC_PATH)/layers/da/layer_da.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/da/layer_crop_scale.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_crop_scale_random.o: $(SRC_PATH)/layers/da/layer_crop_scale_random.cpp $(SRC_PATH)/layers/da/layer_da.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/da/layer_crop_scale_random.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_cutout.o: $(SRC_PATH)/layers/da/layer_cutout.cpp $(SRC_PATH)/layers/da/layer_da.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/da/layer_cutout.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_cutout_random.o: $(SRC_PATH)/layers/da/layer_cutout_random.cpp $(SRC_PATH)/layers/da/layer_da.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/da/layer_cutout_random.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_flip.o: $(SRC_PATH)/layers/da/layer_flip.cpp $(SRC_PATH)/layers/da/layer_da.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/da/layer_flip.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_flip_random.o: $(SRC_PATH)/layers/da/layer_flip_random.cpp $(SRC_PATH)/layers/da/layer_da.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/da/layer_flip_random.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_rotate.o: $(SRC_PATH)/layers/da/layer_rotate.cpp $(SRC_PATH)/layers/da/layer_da.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/da/layer_rotate.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_rotate_random.o: $(SRC_PATH)/layers/da/layer_rotate_random.cpp $(SRC_PATH)/layers/da/layer_da.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/da/layer_rotate_random.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_scale.o: $(SRC_PATH)/layers/da/layer_scale.cpp $(SRC_PATH)/layers/da/layer_da.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/da/layer_scale.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_scale_random.o: $(SRC_PATH)/layers/da/layer_scale_random.cpp $(SRC_PATH)/layers/da/layer_da.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/da/layer_scale_random.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_shift.o: $(SRC_PATH)/layers/da/layer_shift.cpp $(SRC_PATH)/layers/da/layer_da.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/da/layer_shift.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_shift_random.o: $(SRC_PATH)/layers/da/layer_shift_random.cpp $(SRC_PATH)/layers/da/layer_da.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/da/layer_shift_random.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_concat.o: $(SRC_PATH)/layers/merge/layer_concat.cpp $(SRC_PATH)/layers/merge/layer_merge.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_concat.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_matmul.o: $(SRC_PATH)/layers/merge/layer_matmul.cpp $(SRC_PATH)/layers/merge/layer_merge.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_matmul.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_maximum.o: $(SRC_PATH)/layers/merge/layer_maximum.cpp $(SRC_PATH)/layers/merge/layer_merge.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_maximum.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_subtract.o: $(SRC_PATH)/layers/merge/layer_subtract.cpp $(SRC_PATH)/layers/merge/layer_merge.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_subtract.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_add.o: $(SRC_PATH)/layers/merge/layer_add.cpp $(SRC_PATH)/layers/merge/layer_merge.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_add.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_average.o: $(SRC_PATH)/layers/merge/layer_average.cpp $(SRC_PATH)/layers/merge/layer_merge.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_average.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_minimum.o: $(SRC_PATH)/layers/merge/layer_minimum.cpp $(SRC_PATH)/layers/merge/layer_merge.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/merge/layer_minimum.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_div.o: $(SRC_PATH)/layers/operators/layer_div.cpp $(SRC_PATH)/layers/operators/layer_operators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_div.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_diff.o: $(SRC_PATH)/layers/operators/layer_diff.cpp $(SRC_PATH)/layers/operators/layer_operators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_diff.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_pow.o: $(SRC_PATH)/layers/operators/layer_pow.cpp $(SRC_PATH)/layers/operators/layer_operators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_pow.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_exp.o: $(SRC_PATH)/layers/operators/layer_exp.cpp $(SRC_PATH)/layers/operators/layer_operators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_exp.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_sqrt.o: $(SRC_PATH)/layers/operators/layer_sqrt.cpp $(SRC_PATH)/layers/operators/layer_operators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_sqrt.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_log.o: $(SRC_PATH)/layers/operators/layer_log.cpp $(SRC_PATH)/layers/operators/layer_operators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_log.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_log10.o: $(SRC_PATH)/layers/operators/layer_log10.cpp $(SRC_PATH)/layers/operators/layer_operators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_log10.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_operator.o: $(SRC_PATH)/layers/operators/layer_operator.cpp $(SRC_PATH)/layers/operators/layer_operators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_operator.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_abs.o: $(SRC_PATH)/layers/operators/layer_abs.cpp $(SRC_PATH)/layers/operators/layer_operators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_abs.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_sum_op.o: $(SRC_PATH)/layers/operators/layer_sum.cpp $(SRC_PATH)/layers/operators/layer_operators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_sum.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_mult.o: $(SRC_PATH)/layers/operators/layer_mult.cpp $(SRC_PATH)/layers/operators/layer_operators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_mult.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_log2.o: $(SRC_PATH)/layers/operators/layer_log2.cpp $(SRC_PATH)/layers/operators/layer_operators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/operators/layer_log2.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_generator.o: $(SRC_PATH)/layers/generators/layer_generator.cpp $(SRC_PATH)/layers/generators/layer_generators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/generators/layer_generator.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_gauss.o: $(SRC_PATH)/layers/generators/layer_gauss.cpp $(SRC_PATH)/layers/generators/layer_generators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/generators/layer_gauss.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_uniform.o: $(SRC_PATH)/layers/generators/layer_uniform.cpp $(SRC_PATH)/layers/generators/layer_generators.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/generators/layer_uniform.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_gaussian.o: $(SRC_PATH)/layers/noise/layer_gaussian.cpp $(SRC_PATH)/layers/noise/layer_noise.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/noise/layer_gaussian.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_upsampling.o: $(SRC_PATH)/layers/conv/layer_upsampling.cpp $(SRC_PATH)/layers/conv/layer_conv.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/conv/layer_upsampling.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_conv.o: $(SRC_PATH)/layers/conv/layer_conv.cpp $(SRC_PATH)/layers/conv/layer_conv.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/conv/layer_conv.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_transposed_conv.o: $(SRC_PATH)/layers/conv/layer_transposed_conv.cpp $(SRC_PATH)/layers/conv/layer_conv.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/conv/layer_transposed_conv.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_lstm.o: $(SRC_PATH)/layers/recurrent/layer_lstm.cpp $(SRC_PATH)/layers/recurrent/layer_recurrent.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/recurrent/layer_lstm.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_rnn.o: $(SRC_PATH)/layers/recurrent/layer_rnn.cpp $(SRC_PATH)/layers/recurrent/layer_recurrent.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/recurrent/layer_rnn.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_reduction.o: $(SRC_PATH)/layers/reductions/layer_reduction.cpp $(SRC_PATH)/layers/reductions/layer_reductions.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/reductions/layer_reduction.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_var.o: $(SRC_PATH)/layers/reductions/layer_var.cpp $(SRC_PATH)/layers/reductions/layer_reductions.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/reductions/layer_var.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_max.o: $(SRC_PATH)/layers/reductions/layer_max.cpp $(SRC_PATH)/layers/reductions/layer_reductions.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/reductions/layer_max.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_min.o: $(SRC_PATH)/layers/reductions/layer_min.cpp $(SRC_PATH)/layers/reductions/layer_reductions.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/reductions/layer_min.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_sum.o: $(SRC_PATH)/layers/reductions/layer_sum.cpp $(SRC_PATH)/layers/reductions/layer_reductions.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/reductions/layer_sum.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

layer_mean.o: $(SRC_PATH)/layers/reductions/layer_mean.cpp $(SRC_PATH)/layers/reductions/layer_reductions.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/layers/reductions/layer_mean.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

loss_mean_squared_error.o: $(SRC_PATH)/losses/loss_mean_squared_error.cpp $(SRC_PATH)/losses/loss.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/losses/loss_mean_squared_error.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

loss_soft_cross_entropy.o: $(SRC_PATH)/losses/loss_soft_cross_entropy.cpp $(SRC_PATH)/losses/loss.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/losses/loss_soft_cross_entropy.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

loss.o: $(SRC_PATH)/losses/loss.cpp $(SRC_PATH)/losses/loss.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/losses/loss.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

loss_cross_entropy.o: $(SRC_PATH)/losses/loss_cross_entropy.cpp $(SRC_PATH)/losses/loss.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/losses/loss_cross_entropy.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

metric.o: $(SRC_PATH)/metrics/metric.cpp $(SRC_PATH)/metrics/metric.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/metrics/metric.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

metric_categorical_acc.o: $(SRC_PATH)/metrics/metric_categorical_acc.cpp $(SRC_PATH)/metrics/metric.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/metrics/metric_categorical_acc.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

metric_mean_absolute_error.o: $(SRC_PATH)/metrics/metric_mean_absolute_error.cpp $(SRC_PATH)/metrics/metric.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/metrics/metric_mean_absolute_error.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

metric_mean_relative_error.o: $(SRC_PATH)/metrics/metric_mean_relative_error.cpp $(SRC_PATH)/metrics/metric.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/metrics/metric_mean_relative_error.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

metric_mean_squared_error.o: $(SRC_PATH)/metrics/metric_mean_squared_error.cpp $(SRC_PATH)/metrics/metric.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/metrics/metric_mean_squared_error.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

initializer_random_normal.o: $(SRC_PATH)/initializers/initializer_random_normal.cpp $(SRC_PATH)/initializers/initializer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer_random_normal.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

initializer_indentity.o: $(SRC_PATH)/initializers/initializer_indentity.cpp $(SRC_PATH)/initializers/initializer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer_indentity.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

initializer_glorot_normal.o: $(SRC_PATH)/initializers/initializer_glorot_normal.cpp $(SRC_PATH)/initializers/initializer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer_glorot_normal.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

initializer_glorot_uniform.o: $(SRC_PATH)/initializers/initializer_glorot_uniform.cpp $(SRC_PATH)/initializers/initializer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer_glorot_uniform.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

initializer_orthogonal.o: $(SRC_PATH)/initializers/initializer_orthogonal.cpp $(SRC_PATH)/initializers/initializer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer_orthogonal.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

initializer_random_uniform.o: $(SRC_PATH)/initializers/initializer_random_uniform.cpp $(SRC_PATH)/initializers/initializer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer_random_uniform.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

initializer_constant.o: $(SRC_PATH)/initializers/initializer_constant.cpp $(SRC_PATH)/initializers/initializer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer_constant.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

initializer.o: $(SRC_PATH)/initializers/initializer.cpp $(SRC_PATH)/initializers/initializer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/initializers/initializer.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

regularizer_l1.o: $(SRC_PATH)/regularizers/regularizer_l1.cpp $(SRC_PATH)/regularizers/regularizer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/regularizers/regularizer_l1.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

regularizer_l2.o: $(SRC_PATH)/regularizers/regularizer_l2.cpp $(SRC_PATH)/regularizers/regularizer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/regularizers/regularizer_l2.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

regularizer_l1_l2.o: $(SRC_PATH)/regularizers/regularizer_l1_l2.cpp $(SRC_PATH)/regularizers/regularizer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/regularizers/regularizer_l1_l2.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

regularizer.o: $(SRC_PATH)/regularizers/regularizer.cpp $(SRC_PATH)/regularizers/regularizer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_PATH)/regularizers/regularizer.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

#---------------------------------- GPU code
gpu_comparison.o: $(SRC_PATH)/hardware/gpu/gpu_comparison.cu $(SRC_PATH)/hardware/gpu/gpu_hw.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/gpu_comparison.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_core.o: $(SRC_PATH)/hardware/gpu/gpu_core.cu $(SRC_PATH)/hardware/gpu/gpu_hw.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/gpu_core.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_create.o: $(SRC_PATH)/hardware/gpu/gpu_create.cu $(SRC_PATH)/hardware/gpu/gpu_hw.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/gpu_create.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_da.o: $(SRC_PATH)/hardware/gpu/gpu_da.cu $(SRC_PATH)/hardware/gpu/gpu_hw.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/gpu_da.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_generator.o: $(SRC_PATH)/hardware/gpu/gpu_generator.cu $(SRC_PATH)/hardware/gpu/gpu_hw.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/gpu_generator.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_math.o: $(SRC_PATH)/hardware/gpu/gpu_math.cu $(SRC_PATH)/hardware/gpu/gpu_hw.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/gpu_math.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_reduction.o: $(SRC_PATH)/hardware/gpu/gpu_reduction.cu $(SRC_PATH)/hardware/gpu/gpu_hw.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/gpu_reduction.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_tensor.o: $(SRC_PATH)/hardware/gpu/gpu_tensor.cu $(SRC_PATH)/hardware/gpu/gpu_tensor.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/gpu_tensor.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

#---nn
gpu_activations.o: $(SRC_PATH)/hardware/gpu/nn/gpu_activations.cu $(SRC_PATH)/hardware/gpu/nn/gpu_nn.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/nn/gpu_activations.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_conv.o: $(SRC_PATH)/hardware/gpu/nn/gpu_conv.cu $(SRC_PATH)/hardware/gpu/nn/gpu_nn.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/nn/gpu_conv.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_losses.o: $(SRC_PATH)/hardware/gpu/nn/gpu_losses.cu $(SRC_PATH)/hardware/gpu/nn/gpu_nn.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/nn/gpu_losses.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_metrics.o: $(SRC_PATH)/hardware/gpu/nn/gpu_metrics.cu $(SRC_PATH)/hardware/gpu/nn/gpu_nn.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/nn/gpu_metrics.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_pool.o: $(SRC_PATH)/hardware/gpu/nn/gpu_pool.cu $(SRC_PATH)/hardware/gpu/nn/gpu_nn.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/nn/gpu_pool.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_tensor_nn.o: $(SRC_PATH)/hardware/gpu/nn/gpu_tensor_nn.cu $(SRC_PATH)/hardware/gpu/nn/gpu_nn.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/nn/gpu_tensor_nn.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

#--- kernels
gpu_core_kernels.o: $(SRC_PATH)/hardware/gpu/gpu_core_kernels.cu $(SRC_PATH)/hardware/gpu/gpu_kernels.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/gpu_core_kernels.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_create_kernels.o: $(SRC_PATH)/hardware/gpu/gpu_create_kernels.cu $(SRC_PATH)/hardware/gpu/gpu_kernels.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/gpu_create_kernels.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_da_kernels.o: $(SRC_PATH)/hardware/gpu/gpu_da_kernels.cu $(SRC_PATH)/hardware/gpu/gpu_kernels.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/gpu_da_kernels.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_generator_kernels.o: $(SRC_PATH)/hardware/gpu/gpu_generator_kernels.cu $(SRC_PATH)/hardware/gpu/gpu_kernels.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/gpu_generator_kernels.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_math_kernels.o: $(SRC_PATH)/hardware/gpu/gpu_math_kernels.cu $(SRC_PATH)/hardware/gpu/gpu_kernels.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/gpu_math_kernels.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_reduction_kernels.o: $(SRC_PATH)/hardware/gpu/gpu_reduction_kernels.cu $(SRC_PATH)/hardware/gpu/gpu_kernels.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/gpu_reduction_kernels.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

#---nn kernels
gpu_activations_kernels.o: $(SRC_PATH)/hardware/gpu/nn/gpu_activations_kernels.cu $(SRC_PATH)/hardware/gpu/nn/gpu_nn_kernels.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/nn/gpu_activations_kernels.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_conv_kernels.o: $(SRC_PATH)/hardware/gpu/nn/gpu_conv_kernels.cu $(SRC_PATH)/hardware/gpu/nn/gpu_nn_kernels.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/nn/gpu_conv_kernels.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_tensor_nn_kernels.o: $(SRC_PATH)/hardware/gpu/nn/gpu_tensor_nn_kernels.cu $(SRC_PATH)/hardware/gpu/nn/gpu_nn_kernels.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/nn/gpu_tensor_nn_kernels.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_losses_kernels.o: $(SRC_PATH)/hardware/gpu/nn/gpu_losses_kernels.cu $(SRC_PATH)/hardware/gpu/nn/gpu_nn_kernels.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/nn/gpu_losses_kernels.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_metrics_kernels.o: $(SRC_PATH)/hardware/gpu/nn/gpu_metrics_kernels.cu $(SRC_PATH)/hardware/gpu/nn/gpu_nn_kernels.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/nn/gpu_metrics_kernels.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

gpu_pool_kernels.o: $(SRC_PATH)/hardware/gpu/nn/gpu_pool_kernels.cu $(SRC_PATH)/hardware/gpu/nn/gpu_nn_kernels.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c $(SRC_PATH)/hardware/gpu/nn/gpu_pool_kernels.cu -I $(LIBS_PATH) -o $(BUILD_PATH)/$@


#---------------------------------- Tests
aux_tests.o: $(TESTS_PATH)/dev/aux_tests.cpp $(TESTS_PATH)/dev/aux_tests.h
	$(CXX) $(CXXFLAGS) -c $(TESTS_PATH)/dev/aux_tests.cpp -I $(LIBS_PATH) -o $(BUILD_PATH)/$@

#---------------------------------- release

example_ae: $(EXAMPLES_NN_PATH)/eddl_ae.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) $(EXAMPLES_NN_PATH)/eddl_ae.cpp $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS) -I $(LIBS_PATH) -I $(SRC_PATH) -o $(BUILD_PATH)/$@

example_dae: $(EXAMPLES_NN_PATH)/eddl_dae.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) $(EXAMPLES_NN_PATH)/eddl_dae.cpp $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS) -I $(LIBS_PATH) -I $(SRC_PATH) -o $(BUILD_PATH)/$@

example_dae_class: $(EXAMPLES_NN_PATH)/eddl_dae_class.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) $(EXAMPLES_NN_PATH)/eddl_dae_class.cpp $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS) -I $(LIBS_PATH) -I $(SRC_PATH) -o $(BUILD_PATH)/$@

example_load_save: $(EXAMPLES_NN_PATH)/eddl_load_save.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) $(EXAMPLES_NN_PATH)/eddl_load_save.cpp $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS) -I $(LIBS_PATH) -I $(SRC_PATH) -o $(BUILD_PATH)/$@

example_tensor: $(EXAMPLES_T_PATH)/eddl_tensor.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) $(EXAMPLES_T_PATH)/eddl_tensor.cpp $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS) -I $(LIBS_PATH) -I $(SRC_PATH) -o $(BUILD_PATH)/$@

example_train_batch: $(EXAMPLES_NN_PATH)/eddl_train_batch.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) $(EXAMPLES_NN_PATH)/eddl_train_batch.cpp $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS) -I $(LIBS_PATH) -I $(SRC_PATH) -o $(BUILD_PATH)/$@

example_mlp: $(EXAMPLES_NN_PATH)/eddl_mlp.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) $(EXAMPLES_NN_PATH)/eddl_mlp.cpp $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS) -I $(LIBS_PATH) -I $(SRC_PATH) -o $(BUILD_PATH)/$@

example_conv: $(EXAMPLES_NN_PATH)/eddl_conv.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) $(EXAMPLES_NN_PATH)/eddl_conv.cpp $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS) -I $(LIBS_PATH) -I $(SRC_PATH) -o $(BUILD_PATH)/$@

test_main: $(TESTS_PATH)/test_main.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) $(TESTS_PATH)/test_main.cpp $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS) -I $(LIBS_PATH) -I $(SRC_PATH) -o $(BUILD_PATH)/$@


#---------------------------------- Extra options
install:
#	ln -s $(LIBS_PATH) /usr/local/include/
#	ln -s $(SRC_PATH) /usr/local/include/
clean:
	rm *.o *.so example_* *.dot *.pdf
