cmake_minimum_required(VERSION 3.9.2)

###########################################################################
################################ OPTIONS ##################################
###########################################################################


# User options
option(BUILD_SUPERBUILD "Compile using the superbuild system" OFF)
option(BUILD_PROTOBUF "Compile using Protobuf" ON)
option(BUILD_OPENMP "Compile using OpenMP" ON)
option(BUILD_TESTS "Compile tests" ON)
option(BUILD_EXAMPLES "Compile examples" ON)
option(BUILD_SHARED_LIBS "Global flag to cause add_library to create shared libraries if on" ON)
option(BUILD_COVERAGE "Flag to compile for coverage information" OFF)

###########################################################################
################################# BUILD ###################################
###########################################################################

# Build Dependencies
if(BUILD_SUPERBUILD)
    # This is executed at building time: "cmake .."
    PROJECT(eddl-superbuild NONE)

    add_subdirectory(cmake)
endif()
# This is executed at compilation time: "make"
PROJECT(eddl VERSION 0.6 LANGUAGES CXX)


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

# Flags: Global
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Flags: Debugging
if(BUILD_COVERAGE)
    set(COVERAGE_FLAGS ${CMAKE_CXX_FLAGS} "-O0 -g --coverage")

    # Apply coverage flags
    set(CMAKE_EXE_LINKER_FLAGS ${COVERAGE_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${COVERAGE_FLAGS})
endif()

# Flags: Optimization
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Set a default build type if none was specified
set(default_build_type "Release")
#if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
#    set(default_build_type "Debug")
#endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


###########################################################################
############################### WINDOWS ###################################
###########################################################################

## Windows
#if(WIN32)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")   # Disable min/max macros in windef.h
#endif()
#
## Microsoft Visual C++
#if(MSVC)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W1 /MP")   # Enable parallel compilation
#    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
#endif()


###########################################################################
############################## PACKAGES ###################################
###########################################################################

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


###########################################################################
########################## INSTALLATION ###################################
###########################################################################

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
