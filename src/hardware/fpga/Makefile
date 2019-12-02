
CXXFLAGS =-std=c++11 -g
TARGETS := sw_emu
DEVICE := xilinx_u200_xdma_201830_2
TARGET := $(TARGETS)
XCLBIN := kernels/xclbin
XOCC := $(XILINX_SDX)/bin/xocc 
BUILD_DIR := kernels/_x.$(TARGET).$(DEVICE)
BUILD_DIR_tensor = $(BUILD_DIR)/tensor_op
BINARY_CONTAINERS += $(XCLBIN)/tensor_op.$(TARGET).$(DEVICE).xclbin
BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/tensor_op.$(TARGET).$(DEVICE).xo
BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/multitensor_op.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/kernel_sum6.$(TARGET).$(DEVICE).xo
BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/kernel_mult2D.$(TARGET).$(DEVICE).xo
BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/kernel_sum2D_rowwise.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/kernel_cent.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/relu_soft_d.$(TARGET).$(DEVICE).xo
#BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/reduce_sum2D.$(TARGET).$(DEVICE).xo
BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/kernel_accuracy.$(TARGET).$(DEVICE).xo
BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/kernel_total_sum.$(TARGET).$(DEVICE).xo
BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/el_div.$(TARGET).$(DEVICE).xo
BINARY_CONTAINER_tensor_OBJS += $(XCLBIN)/kernel_normalize.$(TARGET).$(DEVICE).xo

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
kernel: $(XCLBIN)/tensor_op.$(TARGET).$(DEVICE).xo  $(BINARY_CONTAINERS)
kernelemu: emuconfig 


#---------------------------------- fpga
#xcl2.o: $(xcl2_SRCS) $(xcl2_HDRS)
#		$(FPGA_CXX) $(FPGA_CXXFLAGS) $(FPGA_LDFLAGS) -c $(xcl2_SRCS) 

#kernel_tests.o: kernel_tests.cpp  
#	$(FPGA_CXX) $(FPGA_CXXFLAGS) $(FPGA_LDFLAGS) -c $(SRC_PATH)/kernel_tests.cpp
 

# Building kernels
$(XCLBIN)/tensor_op.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/tensor_op.cpp 
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k tensor_op -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/multitensor_op.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/multitensor_op.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k multitensor_op -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/kernel_sum6.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/kernel_sum6.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k kernel_sum6 -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/kernel_mult2D.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/kernel_mult2D.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k kernel_mult2D -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/kernel_sum2D_rowwise.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/kernel_sum2D_rowwise.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k kernel_sum2D_rowwise -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/kernel_cent.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/kernel_cent.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k kernel_cent -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/relu_soft_d.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/relu_soft_d.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k relu_soft_d -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/reduce_sum2D.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/reduce_sum2D.cpp
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k reduce_sum2D -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/kernel_accuracy.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/kernel_accuracy.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k kernel_accuracy -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/kernel_total_sum.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/kernel_total_sum.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k kernel_total_sum -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/el_div.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/el_div.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k el_div -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/kernel_normalize.$(TARGET).$(DEVICE).xo: $(SRC_PATH)/kernels/kernel_normalize.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_tensor) -c -k kernel_normalize -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/tensor_op.$(TARGET).$(DEVICE).xclbin: $(BINARY_CONTAINER_tensor_OBJS)
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
	XCL_EMULATION_MODE=$(TARGET) ./test_kernels $(XCLBIN)/tensor_op.$(TARGET).$(DEVICE).xclbin

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
