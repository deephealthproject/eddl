cmake_minimum_required(VERSION 3.9.2)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

## Apple: Don't modify install_name when touching RPATH.
if (APPLE)
    SET(CMAKE_MACOSX_RPATH 1)
endif()

# USER OPTIONS *******************************
set(BUILD_TARGET "GPU" CACHE STRING "Compile library for {CPU, GPU, FPGA}")
option(BUILD_OPENMP "Compile using OpenMP" ON)
option(BUILD_PROTOBUF "Compile using Protobuf" OFF)
option(BUILD_SHARED_LIB "Whether to build a static or dynamic library" ON)
option(BUILD_EXAMPLES "Compile examples" ON)
option(BUILD_TESTS "Compile tests" ON)

# Double checks (restricted args)
set_property(CACHE BUILD_TARGET PROPERTY STRINGS CPU GPU FPGA)

# Device specific setup
string(TOUPPER ${BUILD_TARGET} BUILD_TARGET)  # Detect cpu, Cpu, CPU,...
if(${BUILD_TARGET} STREQUAL "CPU")
elseif(${BUILD_TARGET} STREQUAL "GPU")
elseif(${BUILD_TARGET} STREQUAL "FPGA")
    message(FATAL_ERROR "FPGA not implemented, CMake will exit." )
else()
    message(FATAL_ERROR "Build target not implemented, CMake will exit." )
endif()

# Force default build type to Release
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")

PROJECT(eddl LANGUAGES CXX)

# SET C++ COMPILER STANDARD
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Place binaries and libraries according to GNU standards
include(GNUInstallDirs)

# Flags
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(UNIX)
    option(BUILD_SHARED_LIBS "Build shared libraries(.so)." ON)
else()
    # only support for static build on Windows.
    set(BUILD_SHARED_LIBS OFF)
endif()

# Build Dependencies
add_subdirectory(cmake)
add_subdirectory(src)

# [TESTS] Generate a test executable
if(BUILD_TESTS)
    add_subdirectory(tests)
endif(BUILD_TESTS)


# [EXAMPLES] Generate a few test executables
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif(BUILD_EXAMPLES)


#
## Install
#install(EXPORT CMakeExternalTargets
#        NAMESPACE CMakeExternal::
#        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CMakeExternal
#        COMPONENT Devel)
#include(CMakePackageConfigHelpers)
#configure_package_config_file(cmake/CMakeExternalConfig.cmake.in
#        "${PROJECT_BINARY_DIR}/CMakeExternalConfig.cmake"
#        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CMakeExternal"
#        NO_SET_AND_CHECK_MACRO
#        NO_CHECK_REQUIRED_COMPONENTS_MACRO)
#write_basic_package_version_file(
#        "${PROJECT_BINARY_DIR}/CMakeExternalConfigVersion.cmake"
#        COMPATIBILITY SameMajorVersion)
#install(
#        FILES
#        "${PROJECT_BINARY_DIR}/CMakeExternalConfig.cmake"
#        "${PROJECT_BINARY_DIR}/CMakeExternalConfigVersion.cmake"
#        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CMakeExternal"
#        COMPONENT Devel)