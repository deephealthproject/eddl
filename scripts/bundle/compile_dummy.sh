# Set working path
WORKPATH=/home/salvacarrion/Downloads/eddl_compiled/cpu_x86_64_shared__proto_openmp

# Compiling (include headers: eigen3/Eigen/ and eddl/)
g++ -c -Wall -Werror main.cpp -o main.o \
-I$WORKPATH/include \
-I$WORKPATH/include/eigen3 \

# Linking
g++ main.o -o main \
-L$WORKPATH/lib \
-Wl,-rpath=$WORKPATH/lib \
-leddl