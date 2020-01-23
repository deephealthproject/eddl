cmake_minimum_required(VERSION 3.9.2)

PROJECT(eddl LANGUAGES CXX)

if (${CMAKE_VERSION} VERSION_GREATER "3.12")
    cmake_policy(SET CMP0074 NEW) # Suppress warning for zlib_ROOT variable
endif()

# SET C++ COMPILER STANDARD
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# USER OPTIONS *******************************
set(BUILD_TARGET "CPU" CACHE STRING "Compile library for {CPU, GPU, FPGA}")
option(BUILD_EXAMPLES "Compile examples" ON)
option(BUILD_TESTS "Compile tests" OFF)
option(BUILD_SHARED_LIB "Whether to build a static or dynamic library" OFF)

# Double checks (restricted args)
set_property(CACHE BUILD_TARGET PROPERTY STRINGS CPU GPU FPGA)

# More variables
SET(EXAMPLES_DIR "${PROJECT_SOURCE_DIR}/examples")

# PROPERTIES *******************************
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "targets")

# SET BUILD TYPE (RELEASE/DEBUG)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# SET COMPILER FLAGS
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    #set(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}") // This is a bad idea
    set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# Device specific setup
string(TOUPPER ${BUILD_TARGET} BUILD_TARGET)  # Detect cpu, Cpu, CPU,...
if(${BUILD_TARGET} STREQUAL "CPU")
    message("Compiling for CPU...")
elseif(${BUILD_TARGET} STREQUAL "GPU" OR ${BUILD_TARGET} STREQUAL "CUDA")
    message("Compiling for GPU...")
    set(EDDL_WITH_CUDA ON CACHE BOOL "Enable CUDA support if checked")
elseif(${BUILD_TARGET} STREQUAL "FPGA")
    message("Compiling for FPGA...")
    message(FATAL_ERROR "FPGA not implemented, CMake will exit." )
endif()

# Set default install directories
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if (WIN32)
        set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Installation Directory" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation Directory" FORCE)
    endif()
endif()

# SHOW OPTIONS
message(STATUS "===========================================" )
message(STATUS "===========================================" )
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE} )
message(STATUS "Build target: " ${BUILD_TARGET} )
message(STATUS "Build tests: " ${BUILD_TESTS} )
message(STATUS "Build examples: " ${BUILD_EXAMPLES} )
message(STATUS "Build python: " ${BUILD_PYTHON} )
message(STATUS "CUDA Enabled: " ${USE_CUDA} )
message(STATUS "C++ flags: " ${CMAKE_CXX_FLAGS} )
message(STATUS "CUDA flags: " ${CMAKE_CUDA_FLAGS} )
message(STATUS "===========================================" )
message(STATUS "===========================================" )

# Add CUDA (if possible)
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__")
            set(CMAKE_CUDA_STANDARD 11)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif()
    else()
        message(WARNING "No CUDA support")
    endif()
endif()

# Microsoft Visual C++ - set static/dynamic runtime
if(MSVC)
    if(BUILD_SHARED_LIB)
        message(STATUS "Setting dinamic runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
        if (USE_CUDA AND CMAKE_CUDA_COMPILER)
            foreach(flag_var CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE CMAKE_CUDA_FLAGS_MINSIZEREL CMAKE_CUDA_FLAGS_RELWITHDEBINFO)
                string(REGEX REPLACE "-MT" "-MD" ${flag_var} "${${flag_var}}")
            endforeach(flag_var)
        endif()
    else(BUILD_SHARED_LIB)
        message(STATUS "Setting static runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
        if (USE_CUDA AND CMAKE_CUDA_COMPILER)
            foreach(flag_var CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE CMAKE_CUDA_FLAGS_MINSIZEREL CMAKE_CUDA_FLAGS_RELWITHDEBINFO)
                string(REGEX REPLACE "-MD" "-MT" ${flag_var} "${${flag_var}}")
            endforeach(flag_var)
        endif()
    endif(BUILD_SHARED_LIB)

    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()


# [MACRO] Download submodules
macro(eddl_update_third_party SUBMODULE)
    if(GIT_FOUND AND IS_DIRECTORY "${PROJECT_SOURCE_DIR}/.git")
        # Update submodule as needed
        message(STATUS "${SUBMODULE} update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive third_party/${SUBMODULE}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout ${SUBMODULE} submodule")
        endif()
    endif()
endmacro()


# SET SOURCES *******************************
SET(SOURCE_DIR "src")
file(GLOB CPP_HEADER_FILES_TOP
    ${PROJECT_SOURCE_DIR}/*.cpp
)
file(GLOB CPP_SOURCE_FILES_TOP
    ${PROJECT_SOURCE_DIR}/*.cpp
)
file(GLOB_RECURSE CPP_HEADER_FILES_NESTED
    ${PROJECT_SOURCE_DIR}/src/apis/*.h
    ${PROJECT_SOURCE_DIR}/src/descriptors/*.h
    ${PROJECT_SOURCE_DIR}/src/hardware/cpu/*.h
    ${PROJECT_SOURCE_DIR}/src/initializers/*.h
    ${PROJECT_SOURCE_DIR}/src/layers/*.h
    ${PROJECT_SOURCE_DIR}/src/losses/*.h
    ${PROJECT_SOURCE_DIR}/src/metrics/*.h
    ${PROJECT_SOURCE_DIR}/src/net/*.h
    ${PROJECT_SOURCE_DIR}/src/optimizers/*.h
    ${PROJECT_SOURCE_DIR}/src/regularizers/*.h
    ${PROJECT_SOURCE_DIR}/src/tensor/*.h
)

file(GLOB_RECURSE CPP_SOURCE_FILES_NESTED
    ${PROJECT_SOURCE_DIR}/src/apis/*.cpp
    ${PROJECT_SOURCE_DIR}/src/descriptors/*.cpp
    ${PROJECT_SOURCE_DIR}/src/hardware/cpu/*.cpp
    ${PROJECT_SOURCE_DIR}/src/initializers/*.cpp
    ${PROJECT_SOURCE_DIR}/src/layers/*.cpp
    ${PROJECT_SOURCE_DIR}/src/losses/*.cpp
    ${PROJECT_SOURCE_DIR}/src/metrics/*.cpp
    ${PROJECT_SOURCE_DIR}/src/net/*.cpp
    ${PROJECT_SOURCE_DIR}/src/optimizers/*.cpp
    ${PROJECT_SOURCE_DIR}/src/regularizers/*.cpp
    ${PROJECT_SOURCE_DIR}/src/tensor/*.cpp
)

file(GLOB_RECURSE CUDA_HEADER_FILES
        ${PROJECT_SOURCE_DIR}/src/hardware/gpu/*.h
        )
file(GLOB_RECURSE CUDA_CU_FILES
        ${PROJECT_SOURCE_DIR}/src/hardware/gpu/*.cu
        )

SET(CPP_HEADER_FILES ${CPP_HEADER_FILES_TOP} ${CPP_HEADER_FILES_NESTED})
SET(CPP_SOURCE_FILES ${CPP_SOURCE_FILES_TOP} ${CPP_SOURCE_FILES_NESTED})

SET(CPP_SOURCES ${CPP_HEADER_FILES} ${CPP_SOURCE_FILES})
SET(CUDA_SOURCES ${CUDA_HEADER_FILES} ${CUDA_CU_FILES})

# SET TESTS ****************************
#SET(TEST_DIR "tests")
#SET(TESTS
#    "${TEST_DIR}/test_tensor.cpp"
#    )

# [LIBRARY] EDDL LIBRARY *********************
if(BUILD_SHARED_LIB)
    add_library(eddl SHARED "")
else(BUILD_SHARED_LIB)
    add_library(eddl STATIC "")
endif(BUILD_SHARED_LIB)

target_sources(eddl PRIVATE ${CPP_SOURCES})
target_include_directories(eddl PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
        )

# Link external libraries
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    target_sources(eddl PRIVATE ${CUDA_SOURCES})
    target_include_directories(eddl PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    add_compile_definitions(cGPU)
endif()
target_include_directories(eddl PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen>
        $<INSTALL_INTERFACE:include>
        )

# Target properties
set_target_properties(eddl PROPERTIES
        OUTPUT_NAME_DEBUG "eddld"
        OUTPUT_NAME_RELEASE "eddl"
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Eigen3
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (eddl PUBLIC Eigen3::Eigen)

# OpenMP
find_package(OpenMP)
if (OpenMP_FOUND)
    target_link_libraries(eddl PUBLIC OpenMP::OpenMP_CXX)
    if (${OpenMP_CXX_VERSION_MAJOR})
        set(OpenMP_VERSION_MAJOR ${OpenMP_CXX_VERSION_MAJOR} CACHE INTERNAL "" FORCE)
    endif()
    target_compile_definitions(eddl PUBLIC OpenMP_VERSION_MAJOR=${OpenMP_VERSION_MAJOR})
    message(STATUS "Found OpenMP, version ${OpenMP_VERSION_MAJOR}")

    set(OMP_CANCELLATION true)  # https://www.openmp.org/spec-html/5.0/openmpse59.html
endif()

# CUDA
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    target_link_libraries(eddl
            PRIVATE cublas
            PRIVATE cudart
            PRIVATE curand
            )
endif()

# ZLIB
set(ZLIB_ROOT "" CACHE PATH "Path of zlib install dir. Field not required if detected by CMake.")
find_package(ZLIB REQUIRED)
if (ZLIB_FOUND)
    target_include_directories(eddl PUBLIC ${ZLIB_INCLUDE_DIRS})
    target_link_libraries(eddl PRIVATE ${ZLIB_LIBRARIES})
endif()

# pthreads setup for windows os
# The following commands should be removed when pthread linux
# specific calls will be replaced by thread c++ standard calls
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(EDDL_PTHREADS_INSTALL_PATH "$ENV{PTHREADS_ROOT}" CACHE PATH "Path to the installation of pthreads under Windows")
    if(EDDL_PTHREADS_INSTALL_PATH STREQUAL "")
        message(ERROR ": pthreds install path must be specified under Windows")
    endif()
    target_link_libraries(eddl PRIVATE debug "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVC3d.lib"
            PRIVATE debug "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3d.lib"
            PRIVATE debug "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3d.lib")
    target_link_libraries(eddl PRIVATE optimized "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVC3.lib"
            PRIVATE optimized "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3.lib"
            PRIVATE optimized "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3.lib")
    target_include_directories(eddl PUBLIC "${EDDL_PTHREADS_INSTALL_PATH}/include")
endif()


# pthreads setup for linux os
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    SET(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    SET(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
    target_link_libraries(eddl PRIVATE Threads::Threads)
endif()


# [TESTS] Generate a test executable
if(BUILD_TESTS)
    eddl_update_third_party(googletest)
    set(CMAKE_FOLDER third_party/googletest)
    add_subdirectory(third_party/googletest)
    SET(UNITTEST_NAME "${PROJECT_NAME}_tests")
    set(CMAKE_FOLDER "")
    add_executable(${UNITTEST_NAME} ${TESTS})
    target_include_directories(${UNITTEST_NAME} PUBLIC ${gtest_SOURCE_DIR}/include)
    target_link_libraries(${UNITTEST_NAME} eddl gtest gtest_main)
    add_test(UnitTests ${UNITTEST_NAME})  # To do "make test"
    enable_testing()
endif(BUILD_TESTS)


# [EXAMPLES] Generate a few test executables
if(BUILD_EXAMPLES)
    # EXAMPLES: MNIST ****************************************************
    add_executable(mnist_mlp "${EXAMPLES_DIR}/nn/1_mnist/1_mnist_mlp.cpp")
    target_link_libraries(mnist_mlp eddl)

    add_executable(mnist_mlp_initializers "${EXAMPLES_DIR}/nn/1_mnist/2_mnist_mlp_initializers.cpp")
    target_link_libraries(mnist_mlp_initializers eddl)

    add_executable(mnist_mlp_regularizers "${EXAMPLES_DIR}/nn/1_mnist/3_mnist_mlp_regularizers.cpp")
    target_link_libraries(mnist_mlp_regularizers eddl)

    add_executable(mnist_mlp_da "${EXAMPLES_DIR}/nn/1_mnist/4_mnist_mlp_da.cpp")
    target_link_libraries(mnist_mlp_da eddl)

    add_executable(mnist_mlp_train_batch "${EXAMPLES_DIR}/nn/1_mnist/5_mnist_mlp_train_batch.cpp")
    target_link_libraries(mnist_mlp_train_batch eddl)

    add_executable(mnist_auto_encoder "${EXAMPLES_DIR}/nn/1_mnist/6_mnist_auto_encoder.cpp")
    target_link_libraries(mnist_auto_encoder eddl)

    add_executable(mnist_gan "${EXAMPLES_DIR}/nn/1_mnist/7_mnist_gan.cpp")
    target_link_libraries(mnist_gan eddl)

    add_executable(mnist_wgan "${EXAMPLES_DIR}/nn/1_mnist/8_mnist_wgan.cpp")
    target_link_libraries(mnist_wgan eddl)

    add_executable(mnist_conv "${EXAMPLES_DIR}/nn/1_mnist/9_mnist_conv.cpp")
    target_link_libraries(mnist_conv eddl)

    # EXAMPLES: CIFAR10 ****************************************************
    add_executable(cifar_conv "${EXAMPLES_DIR}/nn/2_cifar10/1_cifar_conv.cpp")
    target_link_libraries(cifar_conv eddl)

    add_executable(cifar_conv_da "${EXAMPLES_DIR}/nn/2_cifar10/2_cifar_conv_da.cpp")
    target_link_libraries(cifar_conv_da eddl)

    add_executable(cifar_vgg16 "${EXAMPLES_DIR}/nn/2_cifar10/3_cifar_vgg16.cpp")
    target_link_libraries(cifar_vgg16 eddl)

    add_executable(cifar_vgg16_bn "${EXAMPLES_DIR}/nn/2_cifar10/4_cifar_vgg16_bn.cpp")
    target_link_libraries(cifar_vgg16_bn eddl)

    add_executable(cifar_vgg16_gn "${EXAMPLES_DIR}/nn/2_cifar10/4_cifar_vgg16_gn.cpp")
    target_link_libraries(cifar_vgg16_gn eddl)

    add_executable(cifar_resnet "${EXAMPLES_DIR}/nn/2_cifar10/5_cifar_resnet.cpp")
    target_link_libraries(cifar_resnet eddl)

    add_executable(cifar_resnet_da_bn "${EXAMPLES_DIR}/nn/2_cifar10/6_cifar_resnet_da_bn.cpp")
    target_link_libraries(cifar_resnet_da_bn eddl)

    add_executable(cifar_resnet50 "${EXAMPLES_DIR}/nn/2_cifar10/7_cifar_resnet50_da_bn.cpp")
    target_link_libraries(cifar_resnet50 eddl)

    # EXAMPLES: Drive segmentation ****************************************************
    add_executable(drive_seg "${EXAMPLES_DIR}/nn/3_drive/1_drive_seg.cpp")
    target_link_libraries(drive_seg eddl)

    # EXAMPLES: Tensor ****************************************************
    add_executable(tensor_manipulation "${EXAMPLES_DIR}/tensor/eddl_tensor.cpp")
    target_link_libraries(tensor_manipulation eddl)

    add_executable(tensor_image "${EXAMPLES_DIR}/tensor/eddl_image.cpp")
    target_link_libraries(tensor_image eddl)

    add_executable(tensor_io "${EXAMPLES_DIR}/tensor/eddl_io.cpp")
    target_link_libraries(tensor_io eddl)
endif(BUILD_EXAMPLES)


# [INSTALL] Install the EDDL library
if (WIN32)
    SET(EDDL_CONFIG_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
else()
    SET(EDDL_CONFIG_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/eddl)
endif()

install(TARGETS eddl
        EXPORT eddlTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )
install(EXPORT eddlTargets DESTINATION ${EDDL_CONFIG_INSTALL_PATH})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/
        DESTINATION include/eddl
        FILES_MATCHING PATTERN "*.h")


## [INSTALL] Install Eigen files
#include("${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/cmake/RegexUtils.cmake")
#test_escape_string_as_regex()
#escape_string_as_regex(ESCAPED_EIGEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/Eigen")
#file(GLOB Eigen_directory_files "${ESCAPED_EIGEN_SOURCE_DIR}/*")
##message(STATUS ${Eigen_directory_files})
#foreach(f ${Eigen_directory_files})
#    if(NOT f MATCHES "\\.txt" AND NOT f MATCHES "${ESCAPED_EIGEN_SOURCE_DIR}/[.].+" AND NOT f MATCHES "${ESCAPED_EIGEN_SOURCE_DIR}/src")
#        list(APPEND Eigen_directory_files_to_install ${f})
#    endif()
#endforeach(f ${Eigen_directory_files})
#install(FILES
#        ${Eigen_directory_files_to_install}
#        DESTINATION "include/Eigen"
#        )
#install(DIRECTORY ${ESCAPED_EIGEN_SOURCE_DIR}/src DESTINATION "include/Eigen" COMPONENT Devel FILES_MATCHING PATTERN "*.h")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/eddlConfig.cmake.in ${CMAKE_BINARY_DIR}/cmake/eddlConfig.cmake @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/cmake/eddlConfig.cmake
        DESTINATION ${EDDL_CONFIG_INSTALL_PATH})

message(STATUS "Install path: ${CMAKE_INSTALL_PREFIX}")
