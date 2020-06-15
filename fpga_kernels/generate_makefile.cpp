#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../include/eddl/hardware/fpga/fpga_enables.h"

int main(int argc, char **argv) {

  // kernels list
  char szKernels[200][50];
  int num_kernels = 0;

#ifdef K_ENABLED_RELU
  strcpy(szKernels[num_kernels++], "relu");
#endif
#ifdef K_ENABLED_D_RELU
  strcpy(szKernels[num_kernels++], "d_relu");
#endif
#ifdef K_ENABLED_ELU
  strcpy(szKernels[num_kernels++], "elu");
#endif
#ifdef K_ENABLED_D_ELU
  strcpy(szKernels[num_kernels++], "elu");
#endif
#ifdef K_ENABLED_MUL_
  strcpy(szKernels[num_kernels++], "mul_");
#endif
#ifdef K_ENABLED_ADD
  strcpy(szKernels[num_kernels++], "add");
#endif
#ifdef K_ENABLED_MUL2D
  strcpy(szKernels[num_kernels++], "mul2d");
#endif
#ifdef K_ENABLED_SUM2D_ROWWISE
  strcpy(szKernels[num_kernels++], "sum2d_rowwise");
#endif  
#ifdef K_ENABLED_SUM
  strcpy(szKernels[num_kernels++], "sum");
#endif
#ifdef K_ENABLED_CENT
  strcpy(szKernels[num_kernels++], "cent");
#endif
#ifdef K_ENABLED_ACCURACY
  strcpy(szKernels[num_kernels++], "accuracy");
#endif  
#ifdef K_ENABLED_REDUCE_SUM2D
  strcpy(szKernels[num_kernels++], "reduce_sum2d");
#endif
#ifdef K_ENABLED_SOFTMAX
  strcpy(szKernels[num_kernels++], "softmax");
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
