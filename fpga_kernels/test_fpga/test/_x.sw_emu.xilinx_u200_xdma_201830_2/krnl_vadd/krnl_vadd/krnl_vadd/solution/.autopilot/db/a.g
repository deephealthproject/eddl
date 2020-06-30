#!/bin/sh
lli=${LLVMINTERP-lli}
exec $lli \
    /home/jorga20j/Vitis_Accel_Examples/cpp_kernels/simple_vadd/_x.sw_emu.xilinx_u200_xdma_201830_2/krnl_vadd/krnl_vadd/krnl_vadd/solution/.autopilot/db/a.g.bc ${1+"$@"}
