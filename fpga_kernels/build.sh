cc generate_makefile.cpp -o generate_makefile
./generate_makefile $1 > Makefile
rm -rf xclbin
make TARGET=sw_emu
