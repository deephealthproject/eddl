The compilation of the EDDLL for FPGA requires compiling the host-side sources using the regular makefile 
and the generation of the kernels/bitstream to program the FPGA. FPGA version of the EDDLL has been adapted to work with 
Xilinx FPĜAs. For the generation of the bitstream Xilinx LICENSES of Vivado tools are required.  

The makefile required to compile the kernels is included in this folder. 

To compile the kernels only type:

make TARGET=hw_emu or make TARGET=sw_emu

to compile them for emulation of to create the FPGA bitstream
 


