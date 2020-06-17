#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../include/eddl/hardware/fpga/fpga_enables.h"

int main(int argc, char **argv) {

  // kernels list
  char szKernels[200][50];
  int num_kernels = 0;

  // Activations
  #ifdef K_ENABLED_RELU
  strcpy(szKernels[num_kernels++], "relu");
  #endif
  #ifdef K_ENABLED_D_RELU
  strcpy(szKernels[num_kernels++], "d_relu");
  #endif
  #ifdef K_ENABLED_THRESHOLDED_RELU
  strcpy(szKernels[num_kernels++], "thresholded_relu");
  #endif
  #ifdef K_ENABLED_D_TRHESHOLDED_RELU
  strcpy(szKernels[num_kernels++], "d_thresholded_relu");
  #endif
  #ifdef K_ENABLED_LEAKY_RELU
  strcpy(szKernels[num_kernels++], "leaky_relu");
  #endif
  #ifdef K_ENABLED_D_LEAKY_RELU
  strcpy(szKernels[num_kernels++], "d_leaky_relu");
  #endif
  #ifdef K_ENABLED_ELU
  strcpy(szKernels[num_kernels++], "elu");
  #endif
  #ifdef K_ENABLED_D_ELU
  strcpy(szKernels[num_kernels++], "d_elu");
  #endif
  #ifdef K_ENABLED_SOFTPLUS
  strcpy(szKernels[num_kernels++], "softplus");
  #endif
  #ifdef K_ENABLED_D_SOFTPLUS
  strcpy(szKernels[num_kernels++], "d_softplus");
  #endif
  #ifdef K_ENABLED_SOFTSIGN
  strcpy(szKernels[num_kernels++], "softsign");
  #endif
  #ifdef K_ENABLED_D_SOFTSIGN
  strcpy(szKernels[num_kernels++], "d_softsign");
  #endif
  #ifdef K_ENABLED_LINEAR
  strcpy(szKernels[num_kernels++], "linear");
  #endif
  #ifdef K_ENABLED_D_LINEAR
  strcpy(szKernels[num_kernels++], "d_linear");
  #endif
  #ifdef K_ENABLED_SIGMOID
  strcpy(szKernels[num_kernels++], "sigmoid");
  #endif
  #ifdef K_ENABLED_D_SIGMOID
  strcpy(szKernels[num_kernels++], "d_sigmoid");
  #endif
  #ifdef K_ENABLED_HARD_SIGMOID
  strcpy(szKernels[num_kernels++], "hard_sigmoid");
  #endif
  #ifdef K_ENABLED_D_HARD_SIGMOID
  strcpy(szKernels[num_kernels++], "d_hard_sigmoid");
  #endif
  #ifdef K_ENABLED_EXP
  strcpy(szKernels[num_kernels++], "exp");
  #endif
  #ifdef K_ENABLED_D_EXP
  strcpy(szKernels[num_kernels++], "d_exp");
  #endif
  #ifdef K_ENABLED_TANH
  strcpy(szKernels[num_kernels++], "tanh");
  #endif
  #ifdef K_ENABLED_D_TANH
  strcpy(szKernels[num_kernels++], "d_tanh");
  #endif
  #ifdef K_ENABLED_SOFTMAX
  strcpy(szKernels[num_kernels++], "softmax");
  #endif
  #ifdef K_ENABLED_D_SOFTMAX
  strcpy(szKernels[num_kernels++], "d_softmax");
  #endif

  // Bn
  #ifdef K_ENABLED_PERMUTE_CHANNELS_LAST
  strcpy(szKernels[num_kernels++], "permute_channels_last");
  #endif
  #ifdef K_ENABLED_PERMUTE_CHANNELS_FIRST
  strcpy(szKernels[num_kernels++], "d_permute_channels_last");
  #endif
  #ifdef K_ENABLED_PERMUTE_BATCH_LAST
  strcpy(szKernels[num_kernels++], "permute_batch_last");
  #endif
  #ifdef K_ENABLED_PERMUTE_BATCH_FIRST
  strcpy(szKernels[num_kernels++], "permute_batch_first");
  #endif

  // Comparison
  #ifdef K_ENABLED_ALL
  strcpy(szKernels[num_kernels++], "all");
  #endif
  #ifdef K_ENABLED_ANY
  strcpy(szKernels[num_kernels++], "any");
  #endif
  #ifdef K_ENABLED_ISFINITE
  strcpy(szKernels[num_kernels++], "isfinite");
  #endif
  #ifdef K_ENABLED_ISINF
  strcpy(szKernels[num_kernels++], "isinf");
  #endif
  #ifdef K_ENABLED_ISNAN
  strcpy(szKernels[num_kernels++], "isnan");
  #endif
  #ifdef K_ENABLED_ISNEGINF
  strcpy(szKernels[num_kernels++], "isneginf");
  #endif
  #ifdef K_ENABLED_ISPOSINF
  strcpy(szKernels[num_kernels++], "isposinf");
  #endif
  #ifdef K_ENABLED_LOGICAL_AND
  strcpy(szKernels[num_kernels++], "logical_and");
  #endif
  #ifdef K_ENABLED_LOGICAL_OR
  strcpy(szKernels[num_kernels++], "logical_or");
  #endif
  #ifdef K_ENABLED_LOGICAL_NOT
  strcpy(szKernels[num_kernels++], "logical_not");
  #endif
  #ifdef K_ENABLED_LOGICAL_XOR
  strcpy(szKernels[num_kernels++], "logigal_xor");
  #endif
  #ifdef K_ENABLED_ALLCLOSE
  strcpy(szKernels[num_kernels++], "allclose");
  #endif
  #ifdef K_ENABLED_ISCLOSE
  strcpy(szKernels[num_kernels++], "isclose");
  #endif
  #ifdef K_ENABLED_GREATER
  strcpy(szKernels[num_kernels++], "greater");
  #endif
  #ifdef K_ENABLED_GREATER_EQUAL
  strcpy(szKernels[num_kernels++], "greater_equal");
  #endif
  #ifdef K_ENABLED_LESS
  strcpy(szKernels[num_kernels++], "less");
  #endif
  #ifdef K_ENABLED_LESS_EQUAL
  strcpy(szKernels[num_kernels++], "less_equal");
  #endif
  #ifdef K_ENABLED_EQUAL
  strcpy(szKernels[num_kernels++], "equal");
  #endif
  #ifdef K_ENABLED_NOT_EQUAL
  strcpy(szKernels[num_kernels++], "not_equal");
  #endif
  #ifdef K_ENABLED_EQUAL2
  strcpy(szKernels[num_kernels++], "equal2");
  #endif

  // Conv
  #ifdef K_ENABLED_IM2COL
  strcpy(szKernels[num_kernels++], "im2col");
  #endif
  #ifdef K_ENABLED_CONV2D
  strcpy(szKernels[num_kernels++], "conv2d");
  #endif


  // Core
  #ifdef K_ENABLED_FILL_
  strcpy(szKernels[num_kernels++], "fill_");
  #endif
  #ifdef K_ENABLED_FILL
  strcpy(szKernels[num_kernels++], "fill");
  #endif
  #ifdef K_ENABLED_SELECT
  strcpy(szKernels[num_kernels++], "select");
  #endif
  #ifdef K_ENABLED_SELECT_BACK
  strcpy(szKernels[num_kernels++], "select_back");
  #endif
  #ifdef K_ENABLED_SET_SELECT
  strcpy(szKernels[num_kernels++], "set_select");
  #endif
  #ifdef K_ENABLED_SET_SELECT_BACK
  strcpy(szKernels[num_kernels++], "set_select_back");
  #endif
  #ifdef K_ENABLED_SELECT2
  strcpy(szKernels[num_kernels++], "select2");
  #endif
  #ifdef K_ENABLED_DESELECT
  strcpy(szKernels[num_kernels++], "deselect");
  #endif
  #ifdef K_ENABLED_CONCAT
  strcpy(szKernels[num_kernels++], "concat");
  #endif

  // Create
  #ifdef K_ENABLED_RANGE
  strcpy(szKernels[num_kernels++], "range");
  #endif
  #ifdef K_ENABLED_EYE
  strcpy(szKernels[num_kernels++], "eye");
  #endif

  // Da
  #ifdef K_ENABLED_SINGLE_SHIFT
  strcpy(szKernels[num_kernels++], "single_shift");
  #endif
  #ifdef K_ENABLED_SINGLE_ROTATE
  strcpy(szKernels[num_kernels++], "single_rotate");
  #endif
  #ifdef K_ENABLED_SINGLE_SCALE
  strcpy(szKernels[num_kernels++], "single_scale");
  #endif
  #ifdef K_ENABLED_SINGLE_FLIP
  strcpy(szKernels[num_kernels++], "single_flip");
  #endif
  #ifdef K_ENABLED_SINGLE_CROP
  strcpy(szKernels[num_kernels++], "single_crop");
  #endif
  #ifdef K_ENABLED_SINGLE_CROP_SCALE
  strcpy(szKernels[num_kernels++], "single_crop_scale");
  #endif
  #ifdef K_ENABLED_CROP_SCALE_RANDOM
  strcpy(szKernels[num_kernels++], "crop_scale_random");
  #endif

  // Generator
  #ifdef K_ENABLED_RAND_UNIFORM
  strcpy(szKernels[num_kernels++], "rand_uniform");
  #endif
  #ifdef K_ENABLED_RAND_SIGNED_UNIFORM
  strcpy(szKernels[num_kernels++], "rand_signed_uniform");
  #endif
  #ifdef K_ENABLED_RAND_BINARY
  strcpy(szKernels[num_kernels++], "rand_binary");
  #endif
  #ifdef K_ENABLED_RAND_NORMAL
  strcpy(szKernels[num_kernels++], "rand_normal");
  #endif

  // Losses
  #ifdef K_ENABLED_CENT
  strcpy(szKernels[num_kernels++], "cent");
  #endif

  // Math
  #ifdef K_ENABLED_ABS_
  strcpy(szKernels[num_kernels++], "abs_");
  #endif
  #ifdef K_ENABLED_ACOS_
  strcpy(szKernels[num_kernels++], "acos_");
  #endif
  #ifdef K_ENABLED_ADD_
  strcpy(szKernels[num_kernels++], "add_");
  #endif
  #ifdef K_ENABLED_ASIN_
  strcpy(szKernels[num_kernels++], "asin_");
  #endif
  #ifdef K_ENABLED_ATAN_
  strcpy(szKernels[num_kernels++], "atan_");
  #endif
  #ifdef K_ENABLED_CEIL_
  strcpy(szKernels[num_kernels++], "ceil_");
  #endif
  #ifdef K_ENABLED_CLAMP_
  strcpy(szKernels[num_kernels++], "clamp_");
  #endif
  #ifdef K_ENABLED_COS_
  strcpy(szKernels[num_kernels++], "cos_");
  #endif
  #ifdef K_ENABLED_COSH_
  strcpy(szKernels[num_kernels++], "cosh_");
  #endif
  #ifdef K_ENABLED_EXP_
  strcpy(szKernels[num_kernels++], "exp_");
  #endif
  #ifdef K_ENABLED_FLOOR_
  strcpy(szKernels[num_kernels++], "floor_");
  #endif
  #ifdef K_ENABLED_INV_
  strcpy(szKernels[num_kernels++], "inv_");
  #endif
  #ifdef K_ENABLED_LOG_
  strcpy(szKernels[num_kernels++], "log_");
  #endif
  #ifdef K_ENABLED_LOG2_
  strcpy(szKernels[num_kernels++], "log2_");
  #endif
  #ifdef K_ENABLED_LOG10_
  strcpy(szKernels[num_kernels++], "log10_");
  #endif
  #ifdef K_ENABLED_LOGN_
  strcpy(szKernels[num_kernels++], "logn_");
  #endif
  #ifdef K_ENABLED_MOD_
  strcpy(szKernels[num_kernels++], "mod_");
  #endif
  #ifdef K_ENABLED_MULT_
  strcpy(szKernels[num_kernels++], "mult_");
  #endif
  #ifdef K_ENABLED_NORMALIZE_
  strcpy(szKernels[num_kernels++], "normalize_");
  #endif
  #ifdef K_ENABLED_POW_
  strcpy(szKernels[num_kernels++], "pow_");
  #endif
  #ifdef K_ENABLED_POWB_
  strcpy(szKernels[num_kernels++], "powb_");
  #endif
  #ifdef K_ENABLED_RECIPROCAL_
  strcpy(szKernels[num_kernels++], "reciprocal_");
  #endif
  #ifdef K_ENABLED_REMAINDER_
  strcpy(szKernels[num_kernels++], "remainder_");
  #endif
  #ifdef K_ENABLED_ROUND_
  strcpy(szKernels[num_kernels++], "round_");
  #endif
  #ifdef K_ENABLED_RSQRT_
  strcpy(szKernels[num_kernels++], "rsqrt_");
  #endif
  #ifdef K_ENABLED_SIGMOID_
  strcpy(szKernels[num_kernels++], "sigmoid_");
  #endif
  #ifdef K_ENABLED_SIGN_
  strcpy(szKernels[num_kernels++], "sign_");
  #endif
  #ifdef K_ENABLED_SIN_
  strcpy(szKernels[num_kernels++], "sin_");
  #endif
  #ifdef K_ENABLED_SINH_
  strcpy(szKernels[num_kernels++], "sinh_");
  #endif
  #ifdef K_ENABLED_SQR_
  strcpy(szKernels[num_kernels++], "sqr_");
  #endif
  #ifdef K_ENABLED_SQRT_
  strcpy(szKernels[num_kernels++], "sqrt_");
  #endif
  #ifdef K_ENABLED_TAN_
  strcpy(szKernels[num_kernels++], "tan_");
  #endif
  #ifdef K_ENABLED_TANH_
  strcpy(szKernels[num_kernels++], "tanh_");
  #endif
  #ifdef K_ENABLED_TRUNC_
  strcpy(szKernels[num_kernels++], "trunc_");
  #endif
  #ifdef K_ENABLED_ADD
  strcpy(szKernels[num_kernels++], "add");
  #endif
  #ifdef K_ENABLED_INC
  strcpy(szKernels[num_kernels++], "inc");
  #endif
  #ifdef K_ENABLED_EL_DIV
  strcpy(szKernels[num_kernels++], "el_div");
  #endif
  #ifdef K_ENABLED_EL_MULT
  strcpy(szKernels[num_kernels++], "el_mult");
  #endif
  #ifdef K_ENABLED_SIGN2
  strcpy(szKernels[num_kernels++], "sign2");
  #endif
  #ifdef K_ENABLED_SUM2D_ROWWISE
  strcpy(szKernels[num_kernels++], "sum2d_rowwise");
  #endif
  #ifdef K_ENABLED_SUM2D_COLWISE
  strcpy(szKernels[num_kernels++], "sum2d_colwise");
  #endif
  #ifdef K_ENABLED_MAX
  strcpy(szKernels[num_kernels++], "max");
  #endif
  #ifdef K_ENABLED_SUM
  strcpy(szKernels[num_kernels++], "sum");
  #endif
  #ifdef K_ENABLED_SUM_ABS
  strcpy(szKernels[num_kernels++], "sum_abs");
  #endif
  #ifdef K_ENABLED_MULT2D
  strcpy(szKernels[num_kernels++], "mult2d");
  #endif 

  // Metrics
  #ifdef K_ENABLED_ACCURACY
  strcpy(szKernels[num_kernels++], "accuracy");
  #endif

  // Pool
  #ifdef K_ENABLED_MPOOL2D
  strcpy(szKernels[num_kernels++], "mpool2d");
  #endif
  #ifdef K_ENABLED_MPOOL2D_BACK
  strcpy(szKernels[num_kernels++], "mpool2d_back");
  #endif
  #ifdef K_ENABLED_AVGPOOL2D
  strcpy(szKernels[num_kernels++], "avgpool2d");
  #endif
  #ifdef K_ENABLED_AVGPOOL2D_BACK
  strcpy(szKernels[num_kernels++], "avgpool2d_back");
  #endif

  // Reduction
  #ifdef K_ENABLED_REDUCE
  strcpy(szKernels[num_kernels++], "reduce");
  #endif
  #ifdef K_ENABLED_REDUCE2
  strcpy(szKernels[num_kernels++], "reduce2");
  #endif
  #ifdef K_ENABLED_REDUCE_OP
  strcpy(szKernels[num_kernels++], "reduce_op");
  #endif
  #ifdef K_ENABLED_OPT2
  strcpy(szKernels[num_kernels++], "opt2");
  #endif
  #ifdef K_ENABLED_REDUCE_SUM2D
  strcpy(szKernels[num_kernels++], "reduce_sum2d");
  #endif
  #ifdef K_ENABLED_REDUCTION
  strcpy(szKernels[num_kernels++], "reduction");
  #endif
  #ifdef K_ENABLED_REDUCTION_BACK
  strcpy(szKernels[num_kernels++], "reduction_back");
  #endif

  // Tensor_nn
  #ifdef K_ENABLED_REPEAT_NN
  strcpy(szKernels[num_kernels++], "repeat_nn");
  #endif
  #ifdef K_ENABLED_D_REPEAT_NN
  strcpy(szKernels[num_kernels++], "d_repeat_nn");
  #endif

  // first part of the makefile
  printf("CXXFLAGS =-std=c++11 -g\n");
  if (argc==1) printf("TARGETS := sw_emu\n"); else printf("TARGETS := %s\n", argv[1]);
  printf("DEVICE := xilinx_u200_xdma_201830_2\n");
  printf("TARGET := $(TARGETS)\n");
  printf("XCLBIN := xclbin\n");
  printf("XCLBIN_NAME := relu\n");
  printf("XOCC := /opt/Xilinx/Vitis/2019.2/bin/v++\n");
  printf("BUILD_DIR := _x.$(TARGET).$(DEVICE)\n");
  printf("BUILD_DIR_tensor = $(BUILD_DIR)/$(XCLBIN_NAME)\n");
  printf("BINARY_CONTAINERS += $(XCLBIN)/$(XCLBIN_NAME).$(TARGET).$(DEVICE).xclbin\n");
  
  // build container
  for (int i=0; i<num_kernels; i++) printf("BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/kernel_%s.$(TARGET).$(DEVICE).xo\n", szKernels[i]);

  printf("xcl2_SRCS:=libs/xcl2.cpp\n");
  printf("xcl2_HDRS:=libs/xcl2.hpp\n");
  printf("xcl2_CXXFLAGS:=-I./libs/\n");
  printf("HOST_SRCS:= test_kernels.cpp $(xcl2_SRCS)\n");
  printf("OPENCL_INCLUDE:= $(XILINX_XRT)/include/\n");
  printf("VIVADO_INCLUDE:= $(XILINX_VIVADO)/include/\n");
  printf("opencl_CXXFLAGS=-I$(OPENCL_INCLUDE) -I$(VIVADO_INCLUDE)\n");
  printf("OPENCL_LIB:=$(XILINX_XRT)/lib/\n");
  printf("opencl_LDFLAGS=-L$(OPENCL_LIB) -lOpenCL -lpthread\n");
  printf("FPGA_CXXFLAGS += $(xcl2_CXXFLAGS)\n");
  printf("FPGA_CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++14\n");
  printf("FPGA_LDFLAGS += $(opencl_LDFLAGS)\n");
  printf("\n");
  printf("# Host compiler global settings\n");
  printf("# CXXFLAGS += -fmessage-length=0\n");
  printf("# LDFLAGS += -lrt -lstdc++\n");
  printf("#\n");
  printf("# Kernel compiler global settings\n");
  printf("CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps\n");
  printf("#\n");
  printf("ifeq ($(XILINX_XRT),)\n");
  printf("$(error Set enviroment variable XILINX_XRT with directory to xilinx runtime)\n");
  printf("endif\n");
  printf("ifeq ($(XILINX_VIVADO),)\n");
  printf("$(error Set enviroment variable XILINX_VIVADO with directory to xilinx Vivado tools)\n");
  printf("endif\n");
  printf("ifeq ($(XILINX_SDX),)\n");
  printf("$(error Set enviroment variable XILINX_SDX with directory to xilinx Vivado tools)\n");
  printf("endif\n");
  printf("\n");
  printf("##CXXFLAGS =-std=c++11 -O3\n");
  printf("CXXFLAGS = $(FPGA_CXXFLAGS) -DcFPGA\n");
  printf("\n");
  printf("FPGA_CXX := $(XILINX_SDX)/bin/xcpp\n");
  printf("FPGA_OBJ = kernel_tests.o xcl2.o\n");
  printf("CXX =$(FPGA_CXX)\n");
  printf("EMCONFIG_DIR = $(XCLBIN)/$(DEVICE)\n");
  printf("CP = cp -rf\n");
  printf("KERNELTEST = test_kernels\n");
  printf("#LDCLFLAGS+=--xp prop:solution.hls_pre_tcl=clock.tcl\n");
  printf("#LDCLFLAGS += --kernel_frequency 65\n");
  printf("SRC_PATH=.\n");
  printf("\n");
  printf("#######################################################################\n");


  for (int i=0; i<num_kernels; i++) printf("k_%s: $(XCLBIN)/kernel_%s.$(TARGET).$(DEVICE).xo $(BINARY_CONTAINERS)\n", szKernels[i], szKernels[i]);
  
  printf("# Building kernels\n");

  for (int i=0; i<num_kernels; i++) {
    printf("$(XCLBIN)/kernel_%s.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernel_%s.cpp\n", szKernels[i], szKernels[i]);
    printf("	mkdir -p $(XCLBIN) \n");
    printf("	$(XOCC) $(CLFLAGS) -c -k k_%s -I'$(<D)' -o'$@' kernel_%s.cpp ;\n", szKernels[i], szKernels[i]);
  }

  printf("\n");
  printf("$(XCLBIN)/$(XCLBIN_NAME).$(TARGET).$(DEVICE).xclbin: $(BINARY_CONTAINER_tensor_OBJS)\n");
  printf("	mkdir -p $(XCLBIN)\n");
  printf("	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -l $(LDCLFLAGS) -o'$@' $(+)\n");
  printf("\n");
  printf("#emulating kernel\n");
  printf("$(KERNELTEST): $(SRC_PATH)/test_kernels.cpp $(xcl2_SRCS) $(xcl2_HDRS)\n");
  printf("	$(FPGA_CXX) $(FPGA_CXXFLAGS) $(SRC_PATH)/test_kernels.cpp $(xcl2_SRCS) $(xcl2_HDRS) -o test_kernels $(FPGA_LDFLAGS)\n");
  printf("\n");
  printf("#emuconfig:$(EMCONFIG_DIR)/emconfig.json\n");
  printf("#$(EMCONFIG_DIR)/emconfig.json:\n");
  printf("emuconfig:\n");
  printf("	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)\n");
  printf("	$(CP) $(EMCONFIG_DIR)/emconfig.json .\n");
  printf("	XCL_EMULATION_MODE=$(TARGET) ./test_kernels $(XCLBIN)/$(XCLBIN_NAME).$(TARGET).$(DEVICE).xclbin\n");
  printf("\n");
  printf("#---------------------------------- release\n");
  printf("#test_eddl: test_eddl.cpp $(FPGA_OBJ)\n");
  printf("	#$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) test_eddl.cpp -o test_eddl $(OBJ) $(GPU_OBJ) $(FPGA_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS) $(FPGA_LDFLAGS)\n");
  printf("#       $(FPGA_CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) $(FPGA_CXXFLAGS) test_eddl.cpp -o test_eddl $(FPGA_OBJ) $(LIBFLAGS) $(FPGA_LDFLAGS)\n");
  printf("\n");
  printf("#-------------------------\n");
  printf("\n");
  printf("install:\n");
  printf("	cp test_eddl /usr/local/bin\n");
  printf("clean:\n");
  printf("	rm *.log\n");
}
