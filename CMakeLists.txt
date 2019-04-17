cmake_minimum_required (VERSION 3.13)

include_directories("C:/Users/Costantino/Desktop/PTHREADS-BUILT/include")
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

project (EDDLL)
add_library(EDDLL "")
add_subdirectory(src)
target_link_libraries (EDDLL "C:/Users/Costantino/Desktop/PTHREADS-BUILT/lib/libpthreadVC3d.lib")

add_executable(MNIST-CONV "")
add_subdirectory(examples/MNIST/CONV)
target_link_libraries (MNIST-CONV EDDLL)
