cmake_minimum_required(VERSION 3.9.2)

PROJECT(eddl LANGUAGES CXX)

# SET C++ COMPILER STANDARD
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Place binaries and libraries according to GNU standards
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# we use this to get code coverage
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

# USER OPTIONS *******************************
set(BUILD_TARGET "GPU" CACHE STRING "Compile library for {CPU, GPU, FPGA}")
option(BUILD_OPENMP "Compile using OpenMP" ON)
option(BUILD_PROTOBUF "Compile using Protobuf" OFF)
#option(BUILD_HPC "Use aggressive flags that might break the code. Use it carefully." OFF) # TODO: Deprecated, not reliable
#option(MKL "Use Intel MKL" OFF)  # TODO: Deprecated, not tested
option(BUILD_SHARED_LIB "Whether to build a static or dynamic library" ON)
option(BUILD_EXAMPLES "Compile examples" ON)
option(BUILD_TESTS "Compile tests" ON)

# Double checks (restricted args)
set_property(CACHE BUILD_TARGET PROPERTY STRINGS CPU GPU FPGA)

# More variables
SET(EXAMPLES_DIR "${PROJECT_SOURCE_DIR}/examples")

# PROPERTIES *******************************
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "targets")

# TODO: Review, Is this needed?
# SET BUILD TYPE (RELEASE/DEBUG)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    #set(CMAKE_BUILD_TYPE "Debug")
endif()

# SET COMPILER FLAGS *******************************
# Read: https://wiki.gentoo.org/wiki/GCC_optimization

# Common
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_FAST_MATH -pipe")

# Windows32
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")   # Disable min/max macros in windef.h
endif()

# Compiler-specific options
if(MSVC)  # Microsoft Visual C++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W1 /MP")   # Enable parallel compilation

else()  # if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fno-finite-math-only -Wall -Wno-missing-braces")

    # Read: https://stackoverflow.com/questions/14492436/g-optimization-beyond-o3-ofast
    if (NOT BUILD_HPC)
        set(CMAKE_CXX_HPC_FLAGS "-mtune=native")
        set(RELEASE_OPT_LEVEL "3")  # O3 might break things
    else()
        # Danger: You can increase the performance but depends on
        set(CMAKE_CXX_HPC_FLAGS "-march=native -fno-signed-zeros -fno-trapping-math -funroll-loops -frename-registers")
        set(RELEASE_OPT_LEVEL "fast")  # -Ofast enables -ffast-math
    endif()

    set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")
    set(CMAKE_CXX_FLAGS_RELEASE "-O${RELEASE_OPT_LEVEL} ${CMAKE_CXX_HPC_FLAGS}")
endif()


# Device specific setup
string(TOUPPER ${BUILD_TARGET} BUILD_TARGET)  # Detect cpu, Cpu, CPU,...
if(${BUILD_TARGET} STREQUAL "CPU")
    #    message("Compiling for CPU...")
elseif(${BUILD_TARGET} STREQUAL "GPU" OR ${BUILD_TARGET} STREQUAL "CUDA")
    #    message("Compiling for GPU...")
    set(USE_CUDA ON CACHE BOOL "Enable CUDA support if checked")
elseif(${BUILD_TARGET} STREQUAL "FPGA")
    #    message("Compiling for FPGA...")
    message(FATAL_ERROR "FPGA not implemented, CMake will exit." )
else()
    message(FATAL_ERROR "Build target not implemented, CMake will exit." )
endif()

# Set default install directories
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if (WIN32)
        set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Installation Directory" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation Directory" FORCE)
    endif()
endif()


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
        set(USE_CUDA OFF)
        set(BUILD_TARGET "CPU")
    endif()
endif()

# TODO: Review, not tested (probably deprecated)
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

# ONNX
if(BUILD_PROTOBUF)
    include(FindProtobuf)
    find_package(Protobuf REQUIRED)
    if(Protobuf_FOUND)
        add_definitions(-DcPROTO)
        #protobuf_generate_cpp(onnx.pb.cc onnx.pb.h onnx.proto) #Generate protobuf files #Doesn't work
        # It seems that in some computers we the path appended to the file
        EXEC_PROGRAM(${PROTOBUF_PROTOC_EXECUTABLE} ARGS --cpp_out ${PROJECT_SOURCE_DIR}/src/serialization/onnx/ --proto_path ${PROJECT_SOURCE_DIR}/src/serialization/onnx/ onnx.proto )
    else()
        SET(BUILD_PROTOBUF OFF)
        message(WARNING "Protobuf not found, ignore if you are not using the ONNX module.")
    endif()
endif()

# TODO: Use another cmakelists?
# C++ files
file(GLOB CPP_HEADER_FILES_TOP
        ${PROJECT_SOURCE_DIR}/src/*.h
        )
file(GLOB CPP_SOURCE_FILES_TOP
        ${PROJECT_SOURCE_DIR}/src/*.cpp
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
SET(CPP_HEADER_FILES ${CPP_HEADER_FILES_TOP} ${CPP_HEADER_FILES_NESTED})
SET(CPP_SOURCE_FILES ${CPP_SOURCE_FILES_TOP} ${CPP_SOURCE_FILES_NESTED})
SET(CPP_SOURCES ${CPP_HEADER_FILES} ${CPP_SOURCE_FILES})


# CUDA files
file(GLOB_RECURSE CUDA_HEADER_FILES
        ${PROJECT_SOURCE_DIR}/src/hardware/gpu/*.h
        )
file(GLOB_RECURSE CUDA_CU_FILES
        ${PROJECT_SOURCE_DIR}/src/hardware/gpu/*.cu
        )
SET(CUDA_SOURCES ${CUDA_HEADER_FILES} ${CUDA_CU_FILES})


# ONNX files
SET(ONNX_SOURCES "")
if(BUILD_PROTOBUF)
    file(GLOB_RECURSE ONNX_HEADER_FILES
            ${PROJECT_SOURCE_DIR}/src/serialization/onnx/*.h
            )
    file(GLOB_RECURSE ONNX_SOURCES_FILES
            ${PROJECT_SOURCE_DIR}/src/serialization/onnx/*.cpp
            ${PROJECT_SOURCE_DIR}/src/serialization/onnx/*.cc
            )
    SET(ONNX_SOURCES ${ONNX_HEADER_FILES} ${ONNX_SOURCES_FILES})
endif()



# [LIBRARY] EDDL LIBRARY *********************
if(BUILD_SHARED_LIB)
    add_library(eddl SHARED "")
else(BUILD_SHARED_LIB)
    add_library(eddl STATIC "")
endif(BUILD_SHARED_LIB)

target_sources(eddl PRIVATE ${CPP_SOURCES} ${ONNX_SOURCES})
target_include_directories(eddl PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
        )

# Target properties
set_target_properties(eddl PROPERTIES
        OUTPUT_NAME_DEBUG "eddld"
        OUTPUT_NAME_RELEASE "eddl"
        )

# Eigen3
find_package(Eigen3 3.3 REQUIRED)
target_link_libraries(eddl PUBLIC Eigen3::Eigen)

# OpenMP
if(BUILD_OPENMP)
    find_package(OpenMP)
    if (OpenMP_FOUND)
        target_link_libraries(eddl PUBLIC OpenMP::OpenMP_CXX)
        if (${OpenMP_CXX_VERSION_MAJOR})
            set(OpenMP_VERSION_MAJOR ${OpenMP_CXX_VERSION_MAJOR} CACHE INTERNAL "" FORCE)
        endif()
        target_compile_definitions(eddl PUBLIC OpenMP_VERSION_MAJOR=${OpenMP_VERSION_MAJOR})
        message(STATUS "Found OpenMP, version ${OpenMP_VERSION_MAJOR}")

        set(OMP_CANCELLATION true)  # https://www.openmp.org/spec-html/5.0/openmpse59.html
    else()
        set(BUILD_OPENMP OFF)
    endif()
endif()

# CUDA
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    target_sources(eddl PRIVATE ${CUDA_SOURCES})
    target_include_directories(eddl PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(eddl
            PRIVATE cublas
            PRIVATE cudart
            PRIVATE curand
            )
    add_definitions(-DcGPU)

    # Target properties
    set_target_properties(eddl PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# ZLIB
set(ZLIB_ROOT "" CACHE PATH "Path of zlib install dir. Field not required if detected by CMake.")
find_package(ZLIB REQUIRED)
if (ZLIB_FOUND)
    target_include_directories(eddl PUBLIC ${ZLIB_INCLUDE_DIRS})
    target_link_libraries(eddl PRIVATE ${ZLIB_LIBRARIES})
endif()


# ONNX
if(BUILD_PROTOBUF)
    include(FindProtobuf)
    find_package(Protobuf REQUIRED)
    if(Protobuf_FOUND)
        add_definitions(-DcPROTO)
        target_include_directories(eddl PUBLIC ${PROTOBUF_INCLUDE_DIR})
        target_link_libraries(eddl PRIVATE ${PROTOBUF_LIBRARIES})
        message("Protobuf path: ${PROTOBUF_LIBRARIES} ${PROTOBUF_INCLUDE_DIR}")
    else()
        message(WARNING "Protobuf not found, ignore if you are not using the ONNX module.")
    endif()
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
    enable_testing()
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})

    add_subdirectory(tests)
endif(BUILD_TESTS)


# [EXAMPLES] Generate a few test executables
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif(BUILD_EXAMPLES)

# TODO: REVIEW installtion paths (potential conflict with GNUInstallDirs)
# [INSTALL] Install the EDDL library
message(STATUS "Install path: ${CMAKE_INSTALL_PREFIX}")

if (WIN32)
    SET(EDDL_CONFIG_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
else()
    SET(EDDL_CONFIG_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/eddl)
endif()

install(TARGETS eddl
        EXPORT eddlTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )
install(EXPORT eddlTargets
        DESTINATION ${EDDL_CONFIG_INSTALL_PATH}
        )
install(DIRECTORY src/
        DESTINATION include/eddl
        FILES_MATCHING PATTERN "*.h")

# This allows to do: "find_package(eddl)"
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/eddlConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/cmake/eddlConfig.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake/eddlConfig.cmake"
        DESTINATION ${EDDL_CONFIG_INSTALL_PATH}
        )

# SHOW OPTIONS
message(STATUS "===========================================" )
message(STATUS "===========================================" )
message(STATUS "Project name: " ${CMAKE_PROJECT_NAME} )
message(STATUS "Prefix path: " ${CMAKE_PREFIX_PATH} )
message(STATUS "Install prefix: " ${CMAKE_INSTALL_PREFIX} )
message(STATUS "Archive output directory: " ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} )
message(STATUS "Library output directory: " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
message(STATUS "Runtime output directory: " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )
message(STATUS "-------------------------------------------" )
message(STATUS "Build target: " ${BUILD_TARGET} )
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE} )
message(STATUS "Build tests: " ${BUILD_TESTS} )
message(STATUS "Build examples: " ${BUILD_EXAMPLES} )
message(STATUS "-------------------------------------------" )
message(STATUS "Use OpenMP: " ${BUILD_OPENMP} )
message(STATUS "Use Protobuf: " ${BUILD_PROTOBUF} )
message(STATUS "Use HPC: " ${BUILD_HPC} )
message(STATUS "Use Intel-MKL: " ${MKL} )
message(STATUS "-------------------------------------------" )
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "C++ flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "C++ flags (release): " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "C++ flags (debug): " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "-------------------------------------------" )
message(STATUS "CUDA Enabled: " ${USE_CUDA} )
if(USE_CUDA)
    message(STATUS "CUDA compiler: " ${CMAKE_CUDA_COMPILER})
    message(STATUS "CUDA flags: " ${CMAKE_CUDA_FLAGS} )
    message(STATUS "CUDA toolkit: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )
endif(USE_CUDA)
message(STATUS "-------------------------------------------" )
if(BUILD_OPENMP)
    message(STATUS "OpenMP version: " ${OpenMP_VERSION_MAJOR} )
    message(STATUS "OpenMP gomp library: " ${OpenMP_gomp_LIBRARY} )
    message(STATUS "OpenMP pthread library: " ${OpenMP_pthread_LIBRARY} )
endif(BUILD_OPENMP)
message(STATUS "-------------------------------------------" )
if(BUILD_PROTOBUF)
    message(STATUS "Protobuf: " ${Protobuf_INCLUDE_DIR} )
    message(STATUS "Protobuf compiler: " ${Protobuf_PROTOC_EXECUTABLE} )
endif(BUILD_PROTOBUF)
message(STATUS "-------------------------------------------" )
message(STATUS "Eigen3: " ${EIGEN3_INCLUDE_DIR} )
message(STATUS "ZLIB: " ${ZLIB_INCLUDE_DIR} )
if(BUILD_TESTS)
message(STATUS "gTests: " ${GTEST_INCLUDE_DIRS} )
endif(BUILD_TESTS)
message(STATUS "===========================================" )
message(STATUS "===========================================" )
