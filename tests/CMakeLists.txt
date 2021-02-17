cmake_minimum_required(VERSION 3.9.2)

project(eddl-tests)

# GTest
enable_testing()
if(GTEST_ROOT)
    # Find libraries (need absolute paths)
    find_library(GTEST_LIBRARY NAMES gtest HINTS ${GTEST_ROOT} PATHS ${GTEST_ROOT} PATH_SUFFIXES "lib" "lib64")
    find_library(GTEST_MAIN_LIBRARY NAMES gtest_main HINTS ${GTEST_ROOT} PATHS ${GTEST_ROOT} PATH_SUFFIXES "lib" "lib64")
    find_library(GTESTD_LIBRARY NAMES gtestd HINTS ${GTEST_ROOT} PATHS ${GTEST_ROOT} PATH_SUFFIXES "lib" "lib64")
    find_library(GTEST_MAIND_LIBRARY NAMES gtest_maind HINTS ${GTEST_ROOT} PATHS ${GTEST_ROOT} PATH_SUFFIXES "lib" "lib64")
else()
    find_package(GTest REQUIRED)
endif()

# Find tests (recursively, from here)
file(GLOB_RECURSE CPP_TESTS_FILES "${PROJECT_SOURCE_DIR}/*" *.{h, cpp})

# Filter ONNX files if they are not needed
if(NOT BUILD_PROTOBUF)
    list(FILTER CPP_TESTS_FILES EXCLUDE REGEX ".*/onnx/*")
endif()

# Build test and target libraries
add_executable(unit_tests ${CPP_TESTS_FILES})
target_include_directories(unit_tests PUBLIC $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>)

# Add libraries
if(MSVC)
    target_link_libraries(unit_tests PUBLIC eddl 
        optimized ${GTEST_LIBRARY} optimized ${GTEST_MAIN_LIBRARY}
        debug ${GTESTD_LIBRARY} debug ${GTEST_MAIND_LIBRARY}
    )
else()
    find_package(Threads REQUIRED)
    target_link_libraries(unit_tests PUBLIC eddl ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
endif()


# CUDA (TEMP! We shouldn't need this)
if(USE_CUDA)
    # Check if cuda is available
    include(CheckLanguage)
    check_language(CUDA)

    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)

        # NVCC needs GCC versions to be less or equal than 8  (GCC < 9.0;)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
                message(FATAL_ERROR "[WARNING] The nvcc compiler in CUDA 10 (or later) does not support gcc versions later than 8 (DETECTED: ${CMAKE_CXX_COMPILER_VERSION}).
        Hint: Use g++-8 (or older); Set other compiler version by using '-D CMAKE_CXX_COMPILER='$(which g++-8)' or creating a symbolic link.")
            endif()
        endif()

        # Set standard CUDA variables
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 11)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif()

        # Target properties
        set_target_properties(unit_tests PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
                )

        # Add source files
        target_sources(unit_tests PRIVATE ${CUDA_HEADERS} ${CUDA_SOURCES})

        # Add includes
        target_include_directories(unit_tests PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

        #        # Find libraries (need absolute paths)
        #        find_library(CUBLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR})
        #        find_library(CUDART_LIBRARY cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR})
        #        find_library(CURAND_LIBRARY curand HINTS ${CUDA_TOOLKIT_ROOT_DIR})
        #        target_link_libraries(unit_tests PRIVATE ${CUBLAS_LIBRARY} ${CUDART_LIBRARY} ${CURAND_LIBRARY})
        target_link_libraries(unit_tests PRIVATE CUDA::cublas CUDA::cudart CUDA::curand)
        if(USE_CUDNN)
            find_library(CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR})
            target_link_libraries(unit_tests PRIVATE ${CUDNN_LIBRARY})
        endif()

        if(APPLE)
            # We need to add the path to the driver (libcuda.dylib) as an rpath,
            # so that the static cuda runtime can find it at runtime.
            set_property(TARGET unit_tests PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        endif()

        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            # Make EDDL works for cuda 7.5
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__")
        endif()
    else()
        message(WARNING "[WARNING] CUDA compiler not found but requested during compilation. (Falling back to: '-D USE_CUDA=OFF')
        Hint: Set the NVCC path using '-D CMAKE_CUDA_COMPILER=path' or creating a symbolic link.")
        SET(BUILD_TARGET "CPU") # Local's scope
    endif()
endif()

##########################################################################
############################# DEFINITIONS ################################
##########################################################################

if(USE_GPU)
    add_definitions(-DcGPU)
endif()
if(USE_CUDNN)
    add_definitions(-DcCUDNN)
endif()
if(USE_FPGA)
    add_definitions(-DcFPGA)
endif()

##########################################################################
############################# EXECUTABLES ################################
##########################################################################

# Add test
add_test(NAME unit_tests COMMAND unit_tests)


##########################################################################
############################### SUMMARY ##################################
##########################################################################

# SHOW OPTIONS
message(STATUS "===========================================" )
message(STATUS "===========================================" )
message(STATUS "GTest root: " ${GTEST_ROOT} )
message(STATUS "GTest include dirs: " ${GTEST_INCLUDE_DIRS} )
message(STATUS "GTest libraries: ${GTEST_LIBRARY}; ${GTEST_MAIN_LIBRARY} ")
#message(STATUS "GTest libraries (debug): ${GTESTD_LIBRARY}; ${GTESTD_MAIN_LIBRARY} ")
message(STATUS "===========================================" )
message(STATUS "===========================================" )
