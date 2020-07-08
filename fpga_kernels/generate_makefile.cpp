#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../include/eddl/hardware/fpga/fpga_enables.h"

int main(int argc, char **argv) {

  // kernels list
  char szKernels[200][50];
  int num_kernels = 0;

  if (argc == 1) {
    printf("Usage: %s <target> <kernel_list>\n", argv[0]);
    exit(1);
  }

  printf("Target: %s\n", argv[1]);

  printf("List of kernels to compile:\n");
  for (int i=2; i<argc; i++) {
    printf(" - %s\n", argv[i]);
    strcpy(szKernels[num_kernels++], argv[i]);
  }

  FILE *fd;

  fd = fopen("Makefile", "w");

  // first part of the makefile
  fprintf(fd, "CXXFLAGS =-std=c++11 -g\n");
  if (argc==1) fprintf(fd, "TARGETS := sw_emu\n"); else fprintf(fd, "TARGETS := %s\n", argv[1]);
  fprintf(fd, "DEVICE := xilinx_u200_xdma_201830_2\n");
  fprintf(fd, "TARGET := $(TARGETS)\n");
  fprintf(fd, "XCLBIN := xclbin\n");
  fprintf(fd, "XCLBIN_NAME := eddl\n");
  fprintf(fd, "XOCC := /opt/Xilinx/Vitis/2019.2/bin/v++\n");
  fprintf(fd, "BUILD_DIR := _x.$(TARGET).$(DEVICE)\n");
  fprintf(fd, "BUILD_DIR_tensor = $(BUILD_DIR)/$(XCLBIN_NAME)\n");
  fprintf(fd, "BINARY_CONTAINERS += $(XCLBIN)/$(XCLBIN_NAME).$(TARGET).$(DEVICE).xclbin\n");
  
  // build container
  for (int i=0; i<num_kernels; i++) fprintf(fd, "BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/kernel_%s.$(TARGET).$(DEVICE).xo\n", szKernels[i]);

  fprintf(fd, "xcl2_SRCS:=libs/xcl2.cpp\n");
  fprintf(fd, "xcl2_HDRS:=libs/xcl2.hpp\n");
  fprintf(fd, "xcl2_CXXFLAGS:=-I./libs/\n");
  fprintf(fd, "HOST_SRCS:= test_kernels.cpp $(xcl2_SRCS)\n");
  fprintf(fd, "OPENCL_INCLUDE:= $(XILINX_XRT)/include/\n");
  fprintf(fd, "VIVADO_INCLUDE:= $(XILINX_VIVADO)/include/\n");
  fprintf(fd, "opencl_CXXFLAGS=-I$(OPENCL_INCLUDE) -I$(VIVADO_INCLUDE)\n");
  fprintf(fd, "OPENCL_LIB:=$(XILINX_XRT)/lib/\n");
  fprintf(fd, "opencl_LDFLAGS=-L$(OPENCL_LIB) -lOpenCL -lpthread\n");
  fprintf(fd, "FPGA_CXXFLAGS += $(xcl2_CXXFLAGS)\n");
  fprintf(fd, "FPGA_CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++14\n");
  fprintf(fd, "FPGA_LDFLAGS += $(opencl_LDFLAGS)\n");
  fprintf(fd, "\n");
  fprintf(fd, "# Host compiler global settings\n");
  fprintf(fd, "# CXXFLAGS += -fmessage-length=0\n");
  fprintf(fd, "# LDFLAGS += -lrt -lstdc++\n");
  fprintf(fd, "LDCLFLAGS += --profile_kernel data:all:all:all:all --profile_kernel stall:all:all:all --profile_kernel exec:all:all:all\n");
  fprintf(fd, "#\n");
  fprintf(fd, "# Kernel compiler global settings\n");
  fprintf(fd, "CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps\n");
  fprintf(fd, "#\n");
  fprintf(fd, "ifeq ($(XILINX_XRT),)\n");
  fprintf(fd, "$(error Set enviroment variable XILINX_XRT with directory to xilinx runtime)\n");
  fprintf(fd, "endif\n");
  fprintf(fd, "ifeq ($(XILINX_VIVADO),)\n");
  fprintf(fd, "$(error Set enviroment variable XILINX_VIVADO with directory to xilinx Vivado tools)\n");
  fprintf(fd, "endif\n");
  fprintf(fd, "ifeq ($(XILINX_SDX),)\n");
  fprintf(fd, "$(error Set enviroment variable XILINX_SDX with directory to xilinx Vivado tools)\n");
  fprintf(fd, "endif\n");
  fprintf(fd, "\n");
  fprintf(fd, "##CXXFLAGS =-std=c++11 -O3\n");
  fprintf(fd, "CXXFLAGS = $(FPGA_CXXFLAGS) -DcFPGA\n");
  fprintf(fd, "\n");
  fprintf(fd, "FPGA_CXX := $(XILINX_SDX)/bin/xcpp\n");
  fprintf(fd, "FPGA_OBJ = kernel_tests.o xcl2.o\n");
  fprintf(fd, "CXX =$(FPGA_CXX)\n");
  fprintf(fd, "EMCONFIG_DIR = $(XCLBIN)/$(DEVICE)\n");
  fprintf(fd, "CP = cp -rf\n");
  fprintf(fd, "KERNELTEST = test_kernels\n");
  fprintf(fd, "#LDCLFLAGS+=--xp prop:solution.hls_pre_tcl=clock.tcl\n");
  fprintf(fd, "#LDCLFLAGS += --kernel_frequency 65\n");
  fprintf(fd, "SRC_PATH=.\n");
  fprintf(fd, "\n");
  fprintf(fd, "#######################################################################\n");


  for (int i=0; i<num_kernels; i++) fprintf(fd, "k_%s: $(XCLBIN)/kernel_%s.$(TARGET).$(DEVICE).xo $(BINARY_CONTAINERS)\n", szKernels[i], szKernels[i]);
  
  fprintf(fd, "# Building kernels\n");

  for (int i=0; i<num_kernels; i++) {
    fprintf(fd, "$(XCLBIN)/kernel_%s.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernel_%s.cpp\n", szKernels[i], szKernels[i]);
    fprintf(fd, "	mkdir -p $(XCLBIN) \n");
    fprintf(fd, "	$(XOCC) $(CLFLAGS) -g -c -k k_%s -I'$(<D)' --profile_kernel stall:all:all:all -o'$@' kernel_%s.cpp ;\n", szKernels[i], szKernels[i]);
  }

  fprintf(fd, "\n");
  fprintf(fd, "$(XCLBIN)/$(XCLBIN_NAME).$(TARGET).$(DEVICE).xclbin: $(BINARY_CONTAINER_tensor_OBJS)\n");
  fprintf(fd, "	mkdir -p $(XCLBIN)\n");
  fprintf(fd, "	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -l $(LDCLFLAGS) -o'$@' $(+)\n");
  fprintf(fd,"\n");
  fprintf(fd, "#emulating kernel\n");
  fprintf(fd, "$(KERNELTEST): $(SRC_PATH)/test_kernels.cpp $(xcl2_SRCS) $(xcl2_HDRS)\n");
  fprintf(fd, "	$(FPGA_CXX) $(FPGA_CXXFLAGS) $(SRC_PATH)/test_kernels.cpp $(xcl2_SRCS) $(xcl2_HDRS) -o test_kernels $(FPGA_LDFLAGS)\n");
  fprintf(fd, "\n");
  fprintf(fd, "#emuconfig:$(EMCONFIG_DIR)/emconfig.json\n");
  fprintf(fd, "#$(EMCONFIG_DIR)/emconfig.json:\n");
  fprintf(fd, "emuconfig:\n");
  fprintf(fd, "	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)\n");
  fprintf(fd, "	$(CP) $(EMCONFIG_DIR)/emconfig.json .\n");
  fprintf(fd, "	XCL_EMULATION_MODE=$(TARGET) ./test_kernels $(XCLBIN)/$(XCLBIN_NAME).$(TARGET).$(DEVICE).xclbin\n");
  fprintf(fd, "\n");
  fprintf(fd, "#---------------------------------- release\n");
  fprintf(fd, "#test_eddl: test_eddl.cpp $(FPGA_OBJ)\n");
  fprintf(fd, "	#$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) test_eddl.cpp -o test_eddl $(OBJ) $(GPU_OBJ) $(FPGA_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS) $(FPGA_LDFLAGS)\n");
  fprintf(fd, "#       $(FPGA_CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) $(FPGA_CXXFLAGS) test_eddl.cpp -o test_eddl $(FPGA_OBJ) $(LIBFLAGS) $(FPGA_LDFLAGS)\n");
  fprintf(fd, "\n");
  fprintf(fd, "#-------------------------\n");
  fprintf(fd, "\n");
  fprintf(fd, "install:\n");
  fprintf(fd, "	cp test_eddl /usr/local/bin\n");
  fprintf(fd, "clean:\n");
  fprintf(fd, "	rm *.log\n");
}
