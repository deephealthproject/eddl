cmake_minimum_required(VERSION 3.9.2)

###########################################################################
################################ OPTIONS ##################################
###########################################################################

# DO NOT TOUCH!!! (Two step building: 1) ON, 2) OFF)
option(USE_SUPERBUILD "Use superbuild system" ON)
option(USE_SYSTEM_DEPENDENCIES "Force the use of system dependencies" OFF)

# User options
option(BUILD_TESTS "Compile tests" ON)
option(BUILD_EXAMPLES "Compile examples" ON)
option(BUILD_SHARED_LIBS "Global flag to cause add_library to create shared libraries if on" ON)

###########################################################################
################################# BUILD ###################################
###########################################################################

# Build Dependencies
if(USE_SUPERBUILD)
    PROJECT(eddl-superbuild NONE)

    add_subdirectory(cmake)
    return() # stop processing this file further
else()
    PROJECT(eddl VERSION 0.4.5 LANGUAGES CXX)
endif()


# SET C++ COMPILER STANDARD
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Place binaries and libraries according to GNU standards
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# Disable in-source builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "FATAL: In-source builds are not allowed. You should create a separate directory for build files.")
endif()

# Flags
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# Build library
add_subdirectory(src)

# Build tests
if(BUILD_TESTS)
    add_subdirectory(tests)
endif(BUILD_TESTS)

# Build examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif(BUILD_EXAMPLES)


