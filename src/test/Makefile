
########################################################################
OBJ =	asin.o alex.o nettable.o utils.o data.o layer.o flayer.o \
	clayer.o player.o catlayer.o net.o cpu_convol.o tensor.o addlayer.o olayer.o
GPU_OBJ =

CXX=g++
CXXFLAGS =  -fopenmp -I Eigen -g

GPU_CXXFLAGS = 
GPU_LIBFLAGS =

LAYERS = layers


ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),install)

## TARGET=1 LINUX
## TARGET=2 MAC
ifdef TARGET
ifeq ($(TARGET),1)
$(info set Linux)
endif
ifeq ($(TARGET),2)
$(info set Mac)
endif
else
$(info ----> DEFAULT: set TARGET LINUX)
CXX = g++
CXXFLAGS = -fopenmp -I Eigen -O3
endif


#### LINUX
ifdef TARGET
ifeq ($(TARGET),1)
# OPT=1: AVX
# OPT=2: SSE
# OPT=3: Nothing
# OPT=4: Debug
CXX = g++
ifdef OPT
ifeq ($(OPT),1)
$(info set AVX)
CXXFLAGS = -fopenmp -I Eigen -march=native -mavx -O3
endif
ifeq ($(OPT),2)
$(info set SSE)
CXXFLAGS = -fopenmp -I Eigen -march=native -msse2 -O3
endif
ifeq ($(OPT),3)
$(info set no opt)
CXXFLAGS = -fopenmp -I Eigen -O3
endif
ifeq ($(OPT),4)
$(info set Debug)
CXXFLAGS =  -fopenmp -I Eigen -g
endif
endif
endif


#### MAC
ifeq ($(TARGET),2)
# OPT=1: AVX
# OPT=2: SSE
# OPT=3: Nothing
# OPT=4: Debug
$(info ##################################################)
$(info Have you install g++ instead of clang?)
$(info if not try: brew install gcc --without-multilib)
$(info ##################################################)
## Put here the g++ version installed
CXX= g++-7
CXXFLAGS = -fopenmp -I . -O3
ifdef OPT
ifeq ($(OPT),1)
$(info set AVX)
CXXFLAGS = -fopenmp -I . -march=native -mavx -O3
endif
ifeq ($(OPT),2)
$(info set SSE)
CXXFLAGS = -fopenmp -I . -march=native -msse2 -O3 
endif
ifeq ($(OPT),3)
$(info set no opt)
CXXFLAGS = -fopenmp -I . -O3
endif
ifeq ($(OPT),4)
$(info set Debug)
CXXFLAGS =  -fopenmp -I . -g
endif
endif
endif

#### POWER
ifeq ($(TARGET),3)
# OPT=1: POWER7
# OPT=2: POWER8
# OPT=3: POWER9
# OPT=4: Debug
$(info #################################)
$(info Perhaps is better to reduce smt:)
$(info sudo ppc64_cpu --smt=1)
$(info Or deactivate it:)
$(info sudo ppc64_cpu --smt=off)
$(info Also play with the cores:)
$(info sudo ppc64_cpu --cores-on=16)
$(info #################################)
CXX= g++
CXXFLAGS = -fopenmp -I . -O3
ifdef OPT
ifeq ($(OPT),1)
$(info set POWER7 VSX)
CXXFLAGS = -fopenmp -I . -mvsx -maltivec -mcpu=power7 -O3
endif
ifeq ($(OPT),2)
$(info set POWER8 VSX)
CXXFLAGS = -fopenmp -I . -mvsx -maltivec -mcpu=power8 -O3
endif
ifeq ($(OPT),3)
$(info set POWER9 VSX)
CXXFLAGS = -fopenmp -I . -mvsx -maltivec -mcpu=power9 -O3
endif
ifeq ($(OPT),4)
$(info set Debug)
CXXFLAGS =  -fopenmp -I . -g
endif
endif
endif

endif




#---------------------------------- Compile for GPU
##GPU FLAGS

ifdef CORE
ifeq ($(CORE),1)
$(info set CORE GPU)
cuda_path := $(LAYERS_CUDA_PATH)
ifeq ($(cuda_path),)
$(error Set enviroment variable LAYERS_CUDA_PATH with directory to cuda)
endif

	CXXFLAGS :=  $(CXXFLAGS) -DfGPU -I $(cuda_path)include/
	GPU_CXXFLAGS := -I $(cuda_path)include/ -L  $(cuda_path)lib64/ -DfGPU
	GPU_OBJ := execution.o cublas_env.o gpu_env.o gpu_kernels.o
        GPU_LIBFLAGS := -lcudart -lcublas -lcurand
	GPU_CXX := $(cuda_path)bin/nvcc
	LAYERS = layers
#ifeq ($(GPU),CUDNN)
#	GPU_LIBFLAGS := $(GPU_LIBFLAGS) -lcudnn
#	CXXFLAGS := $(CXXFLAGS) -DfCUDNN
#endif
endif
else
$(info ----> DEFAULT: set CORE=0 CPU)
endif

endif
endif


all: $(GPU_OBJ) $(OBJ) bin2ascii ascii2bin $(LAYERS)


#---------------------------------- Netparser
C_OPT = -pedantic
C_OBJ = asin.y alex.o asin.o 

nettable.o: nettable.c
	gcc -c nettable.c $(C_OPT)

asin.o:	asin.c
	gcc -c asin.c  $(C_OPT)

alex.o: alex.c asin.c
	gcc -c alex.c  $(C_OPT)

asin.c: asin.y
	bison -o asin.c -d asin.y

alex.c: alex.l 
	flex -oalex.c alex.l


#---------------------------------- layers
utils:	utils.cpp utils.h
	$(CXX) $(CXXFLAGS) -c utils.cpp

tensor: tensor.cpp tensor.h utils.h
	$(CXX) $(CXXFLAGS) -c tensor.cpp

cpu_convol: cpu_convol.cpp cpu_convol.h
	$(CXX) $(CXXFLAGS) -c cpu_convol.cpp

data:	data.cpp data.h utils.h
	$(CXX) $(CXXFLAGS) -c data.cpp

layer:	layer.cpp layer.h utils.h
	$(CXX) $(CXXFLAGS) -c layer.cpp

flayer:	flayer.cpp layer.h utils.h tensor.h
	$(CXX) $(CXXFLAGS) -c flayer.cpp

clayer:	clayer.cpp layer.h utils.h tensor.h
	$(CXX) $(CXXFLAGS) -c clayer.cpp

player:	player.cpp layer.h utils.h 
	$(CXX) $(CXXFLAGS) -c player.cpp

catlayer: catlayer.cpp layer.h utils.h 
	$(CXX) $(CXXFLAGS) -c catlayer.cpp

addlayer: addlayer.cpp layer.h utils.h
	$(CXX) $(CXXFLAGS) -c addlayer.cpp

olayer: olayer.cpp layer.h utils.h
	$(CXX) $(CXXFLAGS) -c olayer.cpp

net:	net.cpp net.h utils.h tensor.h
	$(CXX) $(CXXFLAGS) -c net.cpp

#---------------------------------- gpu code
cublas_env.o: ./gpu/cublas_env.h ./utils.h  ./gpu/execution.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c ./gpu/cublas_env.cu

gpu_env.o: ./gpu/gpu_env.h ./gpu/gpu_kernels.h ./utils.h ./gpu/execution.h
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c ./gpu/gpu_env.cu

gpu_kernels.o:  ./gpu/gpu_kernels.h ./utils.h  
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c ./gpu/gpu_kernels.cu

execution.o: ./gpu/gpu_env.h ./utils.h 
	$(GPU_CXX) $(GPU_CXXFLAGS) $(GPU_LIBFLAGS) -c ./gpu/execution.cu

#-------------------------

ascii2bin: ascii2bin.cpp data.o utils.o $(GPU_OBJ) 
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS)  -o ascii2bin ascii2bin.cpp data.o utils.o $(GPU_OBJ) $(GPU_LIBFLAGS)

bin2ascii: bin2ascii.cpp data.o utils.o
	$(CXX) $(CXXFLAGS)  $(GPU_CXXFLAGS) -o bin2ascii bin2ascii.cpp data.o utils.o  $(GPU_OBJ) $(GPU_LIBFLAGS)

$(LAYERS): main.cpp $(OBJ) $(GPU_OBJ)
	$(CXX) $(CXXFLAGS) $(GPU_CXXFLAGS) -o $(LAYERS) main.cpp $(OBJ) $(GPU_OBJ) $(LIBFLAGS) $(GPU_LIBFLAGS)

#-------------------------

install:
	cp ascii2bin /usr/local/bin
	cp bin2ascii /usr/local/bin
	cp layers /usr/local/bin
clean:
	rm *.o



###########################################################################




















