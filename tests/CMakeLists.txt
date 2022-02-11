cmake_minimum_required(VERSION 3.9.2)

project(eddl-tests)

SET(PROJECT_TESTS_NAME unit_tests)

# GTest
enable_testing()
if(GTEST_ROOT)
    # Find libraries (need absolute paths)
    find_library(GTEST_LIBRARIES NAMES gtest HINTS ${GTEST_ROOT} PATHS ${GTEST_ROOT} PATH_SUFFIXES "lib" "lib64")
    find_library(GTEST_MAIN_LIBRARIES NAMES gtest_main HINTS ${GTEST_ROOT} PATHS ${GTEST_ROOT} PATH_SUFFIXES "lib" "lib64")
    find_library(GTESTD_LIBRARY NAMES gtestd HINTS ${GTEST_ROOT} PATHS ${GTEST_ROOT} PATH_SUFFIXES "lib" "lib64")
    find_library(GTEST_MAIND_LIBRARY NAMES gtest_maind HINTS ${GTEST_ROOT} PATHS ${GTEST_ROOT} PATH_SUFFIXES "lib" "lib64")
    message(STATUS "Using GTest from ${GTEST_ROOT}")
else()
    find_package(GTest REQUIRED)
    message(STATUS "Using GTest from system")
endif()

# Find tests (recursively, from here)
file(GLOB_RECURSE CPP_TESTS_FILES "${PROJECT_SOURCE_DIR}/*" *.{h, cpp})

# Filter ONNX files if they are not needed
if(NOT USE_PROTOBUF)
    list(FILTER CPP_TESTS_FILES EXCLUDE REGEX ".*/onnx/*")
endif()

# Build test and target libraries
add_executable(${PROJECT_TESTS_NAME} ${CPP_TESTS_FILES})
target_include_directories(${PROJECT_TESTS_NAME} PUBLIC $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>)

# Add libraries
if(MSVC)
    target_link_libraries(${PROJECT_TESTS_NAME} PUBLIC eddl 
        optimized ${GTEST_LIBRARIES} optimized ${GTEST_MAIN_LIBRARIES}
        debug ${GTESTD_LIBRARY} debug ${GTEST_MAIND_LIBRARY}
    )
else()
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_TESTS_NAME} PUBLIC eddl ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif()


# CUDA (TEMP! We shouldn't need this)
# CUDA ***************************************
if(USE_CUDA)
    cmake_minimum_required(VERSION 3.17.2)  # Due to CUDAToolkit

    # Check if cuda is available
    include(CheckLanguage)
    check_language(CUDA)

    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)

        #        # NVCC needs GCC versions to be less or equal than 8  (GCC < 9.0;)
        #        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        #            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
        #                message(FATAL_ERROR "[WARNING] The nvcc compiler in CUDA 10 (or later) does not support gcc versions later than 8 (DETECTED: ${CMAKE_CXX_COMPILER_VERSION}).
        #        Hint: Use g++-8 (or older); Set other compiler version by using '-D CMAKE_CXX_COMPILER='$(which g++-8)' or creating a symbolic link.")
        #            endif()
        #        endif()

        # Set standard CUDA variables
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 11)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif()

        # Target properties
        set_target_properties(${PROJECT_TESTS_NAME} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
                )

        # Add source files
        target_sources(${PROJECT_TESTS_NAME} PRIVATE ${CUDA_HEADERS} ${CUDA_SOURCES})

        # Add includes
        target_include_directories(${PROJECT_TESTS_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

        # Add libraries
        target_link_libraries(${PROJECT_TESTS_NAME} PRIVATE CUDA::cublas CUDA::cudart CUDA::curand)
        if(USE_CUDNN)
            if(CUDNN_ROOT_DIR)
                SET(CUDNN_INCLUDE_DIRS ${CUDNN_ROOT_DIR}/include)
                find_library(CUDNN_LIBRARY cudnn HINTS ${CUDNN_ROOT_DIR} PATHS ${CUDNN_ROOT_DIR} PATH_SUFFIXES "lib" "lib64")
            else()
                SET(CUDNN_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
                find_library(CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR} PATHS ${CUDAToolkit_LIBRARY_DIR})
            endif()

            # Check if the library has been found
            if(CUDNN_LIBRARY)
                target_include_directories(${PROJECT_TESTS_NAME} PRIVATE ${CUDNN_INCLUDE_DIRS})
                target_link_libraries(${PROJECT_TESTS_NAME} PRIVATE ${CUDNN_LIBRARY})
            else()
                message(WARNING "[WARNING] CUDNN was not found but requested during compilation. (Falling back to: '-D BUILD_TARGET=GPU')
                Hint: Install CUDNN in the same path as the CUDA Toolkit, or specify the CUDNN path using this flag '-D CUDNN_ROOT_DIR=path'")
                SET(BUILD_TARGET "GPU") # Local's scope
                SET(USE_CUDNN OFF) # Local's scope (disable)
            endif()
        endif()

        if(APPLE)
            # We need to add the path to the driver (libcuda.dylib) as an rpath,
            # so that the static cuda runtime can find it at runtime.
            set_property(TARGET ${PROJECT_TESTS_NAME} PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        endif()

        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            # Make EDDL works for cuda 7.5
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__")
        endif()
    else()
        message(WARNING "[WARNING] CUDA compiler not found but requested during compilation. (Falling back to: '-D BUILD_TARGET=CPU')
        Hint: Set the NVCC path using '-D CMAKE_CUDA_COMPILER=path' or creating a symbolic link.")
        SET(BUILD_TARGET "CPU") # Local's scope
        SET(USE_CUDNN OFF) # Local's scope (disable)
        SET(USE_CUDA OFF) # Local's scope (disable)
    endif()
endif()
SET(USE_CUDA ${USE_CUDA} PARENT_SCOPE)  # Parent's scope
SET(USE_CUDNN ${USE_CUDNN} PARENT_SCOPE)  # Parent's scope

##########################################################################
############################# DEFINITIONS ################################
##########################################################################

if(USE_PROTOBUF)
    add_definitions(-DcPROTO)
endif()
if(USE_CUDA)
    add_definitions(-DcGPU)
endif()
if(USE_CUDNN)
    add_definitions(-DcCUDNN)
endif()
if(USE_FPGA_INTEL)
    add_definitions(-DcFPGA)
    add_definitions(-DcFPGA_VENDOR_INTEL)
endif()
if(USE_FPGA_XILINX)
    add_definitions(-DcFPGA)
    add_definitions(-DcFPGA_VENDOR_XILINX)
endif

##########################################################################
############################# EXECUTABLES ################################
##########################################################################

# Add test
add_test(NAME ${PROJECT_TESTS_NAME} COMMAND ${PROJECT_TESTS_NAME})


##########################################################################
############################### SUMMARY ##################################
##########################################################################

# SHOW OPTIONS
message(STATUS "===========================================" )
message(STATUS "===========================================" )
message(STATUS "GTest root: " ${GTEST_ROOT} )
message(STATUS "GTest include dirs: " ${GTEST_INCLUDE_DIRS} )
message(STATUS "GTest libraries: ${GTEST_LIBRARIES}; ${GTEST_MAIN_LIBRARIES} ")
#message(STATUS "GTest libraries (debug): ${GTESTD_LIBRARY}; ${GTESTD_MAIN_LIBRARY} ")
message(STATUS "===========================================" )
message(STATUS "===========================================" )
