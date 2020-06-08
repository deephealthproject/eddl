
CXXFLAGS =-std=c++11 -g
TARGETS := sw_emu
DEVICE := xilinx_u200_xdma_201830_2
TARGET := $(TARGETS)
XCLBIN := kernels/xclbin
XCLBIN_NAME := relu
XOCC := /opt/Xilinx/Vitis/2019.2/bin/v++
BUILD_DIR := kernels/_x.$(TARGET).$(DEVICE)
BUILD_DIR_tensor = $(BUILD_DIR)/$(XCLBIN_NAME)
BINARY_CONTAINERS += $(XCLBIN)/$(XCLBIN_NAME).$(TARGET).$(DEVICE).xclbin

BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/activations.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/bn.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/comparison.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/conv.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/core.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/create.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/da.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/generator.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/losses.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/math.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/metrics.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/pool.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/reduction.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/tensor_nn.$(TARGET).$(DEVICE).xo#


xcl2_SRCS:=libs/xcl2.cpp
xcl2_HDRS:=libs/xcl2.hpp
xcl2_CXXFLAGS:=-I./libs/
HOST_SRCS:= test_kernels.cpp $(xcl2_SRCS)
OPENCL_INCLUDE:= $(XILINX_XRT)/include/
VIVADO_INCLUDE:= $(XILINX_VIVADO)/include/
opencl_CXXFLAGS=-I$(OPENCL_INCLUDE) -I$(VIVADO_INCLUDE)
OPENCL_LIB:=$(XILINX_XRT)/lib/
opencl_LDFLAGS=-L$(OPENCL_LIB) -lOpenCL -lpthread
FPGA_CXXFLAGS += $(xcl2_CXXFLAGS)
FPGA_CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++14
FPGA_LDFLAGS += $(opencl_LDFLAGS)


# Host compiler global settings
# CXXFLAGS += -fmessage-length=0
# LDFLAGS += -lrt -lstdc++
#
# Kernel compiler global settings
CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps
#

ifeq ($(XILINX_XRT),)
$(error Set enviroment variable XILINX_XRT with directory to xilinx runtime)
endif
ifeq ($(XILINX_VIVADO),)
$(error Set enviroment variable XILINX_VIVADO with directory to xilinx Vivado tools)
endif
ifeq ($(XILINX_SDX),)
$(error Set enviroment variable XILINX_SDX with directory to xilinx Vivado tools)
endif

##CXXFLAGS =-std=c++11 -O3
CXXFLAGS = $(FPGA_CXXFLAGS) -DcFPGA

FPGA_CXX := $(XILINX_SDX)/bin/xcpp
FPGA_OBJ = kernel_tests.o xcl2.o
CXX =$(FPGA_CXX)
EMCONFIG_DIR = $(XCLBIN)/$(DEVICE)
CP = cp -rf
KERNELTEST = test_kernels
#LDCLFLAGS+=--xp prop:solution.hls_pre_tcl=clock.tcl
#LDCLFLAGS += --kernel_frequency 65
SRC_PATH=.

#######################################################################

#test: $(FPGA_OBJ)
kernel: $(XCLBIN)/activations.$(TARGET).$(DEVICE).xo  $(BINARY_CONTAINERS)
kernelemu: emuconfig


#---------------------------------- fpga
#xcl2.o: $(xcl2_SRCS) $(xcl2_HDRS)
#		$(FPGA_CXX) $(FPGA_CXXFLAGS) $(FPGA_LDFLAGS) -c $(xcl2_SRCS)

#kernel_tests.o: kernel_tests.cpp
#	$(FPGA_CXX) $(FPGA_CXXFLAGS) $(FPGA_LDFLAGS) -c $(SRC_PATH)/kernel_tests.cpp


# Building kernels
$(XCLBIN)/activations.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/activations.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k k_relu -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/bn.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/bn.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k bn -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/comparison.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/comparison.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k comparison -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/conv.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/conv.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k conv -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/core.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/core.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k core -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/create.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/create.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k create -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/da.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/da.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k da -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/generator.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/generator.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k generator -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/losses.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/losses.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k losses -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/math.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/math.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k math -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/metrics.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/metrics.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k metrics -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/pool.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/pool.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k pool -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/reduction.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/reduction.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k reduction -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/tensor_nn.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/tensor_nn.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k tensor_nn -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/$(XCLBIN_NAME).$(TARGET).$(DEVICE).xclbin: $(BINARY_CONTAINER_tensor_OBJS)
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -l $(LDCLFLAGS) -o'$@' $(+)



#emulating kernel
$(KERNELTEST): $(SRC_PATH)/test_kernels.cpp $(xcl2_SRCS) $(xcl2_HDRS)
	 $(FPGA_CXX) $(FPGA_CXXFLAGS) $(SRC_PATH)/test_kernels.cpp $(xcl2_SRCS) $(xcl2_HDRS) -o test_kernels $(FPGA_LDFLAGS)

#emuconfig:$(EMCONFIG_DIR)/emconfig.json
#$(EMCONFIG_DIR)/emconfig.json:
emuconfig:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) ./test_kernels $(XCLBIN)/$(XCLBIN_NAME).$(TARGET).$(DEVICE).xclbin

#---------------------------------- release
#test_eddl: test_eddl.cpp $(FPGA_OBJ)
	#$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) test_eddl.cpp -o test_eddl $(OBJ) $(GPU_OBJ) $(FPGA_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS) $(FPGA_LDFLAGS)
#	$(FPGA_CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) $(FPGA_CXXFLAGS) test_eddl.cpp -o test_eddl $(FPGA_OBJ) $(LIBFLAGS) $(FPGA_LDFLAGS)

#-------------------------

install:
	cp test_eddl /usr/local/bin
clean:
	rm *.log






###########################################################################
