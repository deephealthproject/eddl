# list of kernel test to compile
LIST ?=conv2D_K3x3_S1x1_P1x1_BS1

# default target
all build clean cleanall: KERNELS

KERNELS : 
	for krnl in $(LIST); do \
		$(info Launch Makefile to generate test for kernel $(krnl)) \
		$(MAKE) -C ./src -e KNAME=$$krnl $(MAKECMDGOALS) ; \
	done
# build any target by forwarding to $(dirs) rule
#% : $(DIRS) ;

.PHONY: KERNELS all build clean cleanall


$(info all done)

