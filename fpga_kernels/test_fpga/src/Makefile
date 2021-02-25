.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO)  "  make test DEVICE=<FPGA platform>"
	$(ECHO)  "     Command to run the application. This is same as 'check' target but does not have any makefile dependency."
	$(ECHO)  ""
	$(ECHO) "  make sd_card TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to prepare sd_card files."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make check TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""


KNAME ?= DUMMY

KRNL_DIR  ?= ./../..
KRNL_FUNC ?= k_$(KNAME)
KRNL_NAME ?= kernel_$(KNAME)
KRNL_SRCS ?= $(KRNL_DIR)/kernel_$(KNAME).cpp
KRNL_FILE := kernel_$(KNAME).cpp
TEST_SRCS ?= test_$(KNAME).cpp

DEVICE ?= xilinx_u200_xdma_201830_2
TARGET ?= sw_emu
HOST_ARCH ?= x86
SYSROOT ?= 

$(info )
$(info )
$(info Running Makefile for KERNEL $(KNAME)  DEVICE $(DEVICE) TARGET $(TARGET) )
$(info )

BASE_DIR = ./..

# Points to top directory of Git repository
COMMON_REPO = $(BASE_DIR)
PWD = $(shell readlink -f .)
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))


include $(BASE_DIR)/utils.mk

XSA := $(call device2xsa, $(DEVICE))
TEMP_DIR := $(BASE_DIR)/_x.$(TARGET).$(XSA)
BUILD_DIR := $(BASE_DIR)/build_dir.$(TARGET).$(XSA)

# SoC variables
RUN_APP_SCRIPT = run_app.sh
PACKAGE_OUT = package.$(TARGET)

LAUNCH_EMULATOR = $(PACKAGE_OUT)/launch_$(TARGET).sh
RESULT_STRING = TEST PASSED

VPP := v++
SDCARD := sd_card

#Include Libraries
include $(ABS_COMMON_REPO)/common/includes/opencl/opencl.mk
include $(ABS_COMMON_REPO)/common/includes/xcl2/xcl2.mk
CXXFLAGS += $(xcl2_CXXFLAGS)
LDFLAGS += $(xcl2_LDFLAGS)
HOST_SRCS += $(xcl2_SRCS)
CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++11 
LDFLAGS += $(opencl_LDFLAGS)
INCL_DIR := -I$(KRNL_DIR)

HOST_SRCS += $(TEST_SRCS)

# Host compiler global settings
CXXFLAGS += -fmessage-length=0
LDFLAGS += -lrt -lstdc++ 


ifneq ($(HOST_ARCH), x86)
	LDFLAGS += --sysroot=$(SYSROOT)
endif

# Kernel compiler global settings
CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps
ifneq ($(TARGET), hw)
	CLFLAGS += -g
endif

EXECUTABLE := $(TEST_SRCS:%.cpp=%)
#$(info "EXECUTABLE is $(EXECUTABLE)" ) 
KRNL_XCLBIN = $(KRNL_NAME:%=%.xclbin)
KRNL_OBJ = $(KRNL_NAME:%=%.xo)

CMD_ARGS = $(BUILD_DIR)/$(KRNL_XCLBIN)
EMCONFIG_DIR = $(TEMP_DIR)
EMU_DIR = $(SDCARD)/data/emulation

BINARY_CONTAINERS += $(BUILD_DIR)/$(KRNL_XCLBIN)
BINARY_CONTAINER_kernel_OBJS += $(TEMP_DIR)/$(KRNL_OBJ)

CP = cp -rf

.PHONY: all clean cleanall docs emconfig
all: check-devices $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig sd_card

.PHONY: exe
exe: $(EXECUTABLE)

.PHONY: build
build: check-vitis $(BINARY_CONTAINERS)


# Building kernel
$(TEMP_DIR)/$(KRNL_OBJ): $(KRNL_SRCS)
	mkdir -p $(TEMP_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(TEMP_DIR) -c -k $(KRNL_FUNC) -I'$(<D)' -I../ -o'$@' '$<'
$(BUILD_DIR)/$(KRNL_XCLBIN): $(BINARY_CONTAINER_kernel_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(BUILD_DIR) -l $(LDCLFLAGS) -o'$@' $(+) #--config ./cfg_connectivity.ini

# Building Host
$(EXECUTABLE): check-xrt $(HOST_SRCS) $(HOST_HDRS)
	$(CXX) $(CXXFLAGS) $(HOST_SRCS) $(HOST_HDRS) -o '$(BASE_DIR)/$@' $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

check: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) $(BUILD_DIR)/$(KRNL_XCLBIN)
else
	$(ABS_COMMON_REPO)/common/utility/run_emulation.pl "./${LAUNCH_EMULATOR} | tee run_app.log" "./${RUN_APP_SCRIPT} $(TARGET)" "${RESULT_STRING}" "7"
endif
else
ifeq ($(HOST_ARCH), x86)
	./$(EXECUTABLE) $(BUILD_DIR)/$(KRNL_XCLBIN)
endif
endif

.PHONY: test
test: $(EXECUTABLE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) $(BUILD_DIR)/$(KRNL_XCLBIN)
else
	$(ABS_COMMON_REPO)/common/utility/run_emulation.pl "./${LAUNCH_EMULATOR} | tee embedded_run.log" "./${RUN_APP_SCRIPT} $(TARGET)" "${RESULT_STRING}" "7"
endif
else
ifeq ($(HOST_ARCH), x86)
	./$(EXECUTABLE) $(BUILD_DIR)/$(KRNL_XCLBIN)
else
	$(ECHO) "Please copy the content of sd_card folder and data to an SD Card and run on the board"
endif
endif


sd_card: gen_run_app
ifneq ($(HOST_ARCH), x86)
	$(VPP) -t $(TARGET) --platform $(DEVICE) -p $(BUILD_DIR)/$(KRNL_XCLBIN) --package.out_dir $(PACKAGE_OUT) --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 --package.sd_file $(SD_IMAGE_FILE) --package.sd_file xrt.ini --package.sd_file $(RUN_APP_SCRIPT) --package.sd_file $(EXECUTABLE) -o $(KRNL_XCLBIN)
endif

# Cleaning stuff
clean:
	-$(RMDIR) "$(BASE_DIR)/$(EXECUTABLE)" "$(BASE_DIR)/$(XCLBIN)/*sw_emu*"  "$(BASE_DIR)/$(XCLBIN)/*hw_emu*"
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv $(BASE_DIR)/*.rpt $(BASE_DIR)/*.csv 
	-$(RMDIR) /*.ll *v++* .Xil $(BASE_DIR)/emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) $(BASE_DIR)/build_dir* sd_card*
	-$(RMDIR) package.*
	-$(RMDIR) $(BASE_DIR)/_x* $(BASE_DIR)/*xclbin.run_summary qemu-memory-_* emulation/ _vimage/ pl* start_simulation.sh $(BASE_DIR)/*.xclbin
	-$(RMDIR) cmake_build/CMakeCache.txt cmake_build/CMakeFiles cmake_build/Makefile cmake_build/cmake_install.cmake 
	
