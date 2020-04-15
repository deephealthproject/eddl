cmake_minimum_required(VERSION 3.9.2)

###########################################################################
################################ OPTIONS ##################################
###########################################################################

# DO NOT TOUCH!!! (Two step building: 1) ON, 2) OFF)
option(USE_SUPERBUILD "Use superbuild system" ON)

# User options
option(BUILD_TESTS "Compile tests" ON)
option(BUILD_EXAMPLES "Compile examples" ON)
option(BUILD_SHARED_LIBS "Global flag to cause add_library to create shared libraries if on" ON)
option(BUILD_COVERAGE "Flag to compile for coverage information" OFF)

###########################################################################
################################# BUILD ###################################
###########################################################################

# Build Dependencies
if(USE_SUPERBUILD)
    # This is executed at building time: "cmake .."
    PROJECT(eddl-superbuild NONE)

    add_subdirectory(cmake)
#    return() # stop processing this file further
#else()
endif()
# This is executed at compilation time: "make"
PROJECT(eddl VERSION 0.4.6 LANGUAGES CXX)


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

# Coverage
if(BUILD_COVERAGE)
set(COVERAGE_FLAGS ${CMAKE_CXX_FLAGS} "-O0 -g --coverage")
set(CMAKE_EXE_LINKER_FLAGS ${COVERAGE_FLAGS})
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${COVERAGE_FLAGS})
endif()

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

# Install
install(EXPORT EDDLTargets
        NAMESPACE EDDL::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/eddl
        COMPONENT devel)
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/EDDLConfig.cmake.in
        "${PROJECT_BINARY_DIR}/cmake/eddl/EDDLConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/eddl"
        NO_SET_AND_CHECK_MACRO
        NO_CHECK_REQUIRED_COMPONENTS_MACRO)
write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/cmake/eddl/EDDLConfigVersion.cmake"
        COMPATIBILITY SameMajorVersion)
install(
        FILES
        "${PROJECT_BINARY_DIR}/cmake/eddl/EDDLConfig.cmake"
        "${PROJECT_BINARY_DIR}/cmake/eddl/EDDLConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/eddl"
        COMPONENT devel)
