#!/bin/sh
lli=${LLVMINTERP-lli}
exec $lli \
    /home/jorga20j/test_fpga/simple_vadd/_x.sw_emu.xilinx_u200_xdma_201830_2/kernel_relu/k_relu/k_relu/solution/.autopilot/db/a.g.bc ${1+"$@"}
