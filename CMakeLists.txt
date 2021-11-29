cmake_minimum_required(VERSION 3.9.2)

###########################################################################
################################ OPTIONS ##################################
###########################################################################


# User options
option(BUILD_SUPERBUILD "Compile using the superbuild system" OFF)
option(BUILD_PROTOBUF "Compile using Protobuf" ON)
option(BUILD_OPENMP "Compile using OpenMP" ON)
option(BUILD_HPC "Compile using aggressive flags for performance" ON)
option(BUILD_TESTS "Compile tests (HPC needs to be disabled)" OFF)  # Disable HPC to pass tests (there are numerical errors)
option(BUILD_EXAMPLES "Compile examples" ON)
option(BUILD_DIST "Compile for a distributed execution" OFF)
option(BUILD_RUNTIME "Compile runtime" OFF)
option(BUILD_COVERAGE "Flag to compile for coverage information" OFF)
option(BUILD_SANITIZERS "Flag to compile with sanitizers information" OFF)

if(WIN32)
    option(BUILD_SHARED_LIBS "Global flag to cause add_library to create shared libraries if on" OFF)
else()
    option(BUILD_SHARED_LIBS "Global flag to cause add_library to create shared libraries if on" ON)
endif()

###########################################################################
################################# BUILD ###################################
###########################################################################

set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)

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

# Build Dependencies
if(BUILD_SUPERBUILD)
    # This is executed at building time: "cmake .."
    PROJECT(eddl-superbuild NONE)

    add_subdirectory(cmake)
endif()
# This is executed at compilation time: "make"
PROJECT(eddl VERSION 1.0 LANGUAGES CXX)


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


if(UNIX)
    # Flags: Optimization
    # READ THIS: https://wiki.gentoo.org/wiki/GCC_optimization

    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")

    # Flags: Sanitizers
    # For the visualized output to be available, switch to Clang at least 3.8.0 or GCC at least 5.0.0
    if(BUILD_SANITIZERS)
        set(CMAKE_CXX_FLAGS_DEBUG "-O1 -g")

        #For more info: https://www.jetbrains.com/help/clion/google-sanitizers.html
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")  # address, leak, thread, undefined, memory
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize-recover=address")  # address, leak, thread, undefined, memory
    endif()

    # Code coverage
    # Read: https://www.jetbrains.com/help/clion/code-coverage-clion.html
    if(BUILD_COVERAGE)
        set(COVERAGE_FLAGS ${CMAKE_CXX_FLAGS} "--coverage")

        # Apply coverage flags
        set(CMAKE_EXE_LINKER_FLAGS ${COVERAGE_FLAGS})
        set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${COVERAGE_FLAGS})
    endif()

    if(BUILD_HPC)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -msse -mfpmath=sse -ffast-math")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ftree-vectorize")
    endif()
endif()



###########################################################################
############################### WINDOWS ###################################
###########################################################################

# Windows
if(WIN32)
    add_compile_definitions(
            NOMINMAX # Disable min/max macros in windef.h
            _CRT_SECURE_NO_WARNINGS # Eliminate deprecation warnings less secure functions
    )
    # Microsoft Visual C++
    add_compile_options(
            $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:-W1> # Display level 1 (severe) warnings
            $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:-MP> # Enable parallel compilation
    )
endif()


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

# Build runtime
if(BUILD_RUNTIME)
    add_subdirectory(runtime)
endif(BUILD_RUNTIME)


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



###########################################################################
############################### WARNINGS ##################################
###########################################################################

# Issues with HPC and logic functions
if(BUILD_HPC)
    message(WARNING "[WARNING] The 'BUILD_HPC' flag is enabled and might not work on a different CPU.
If you intend to compile the library for different CPUs (or for a docker image), you should disable it: '-D BUILD_HPC=OFF'. This happens because it makes use of native compiling flags.")
endif()

# Issues with HPC and logic functions
if(BUILD_TESTS AND BUILD_HPC)
    message(WARNING "[WARNING] Some logic functions are not compatible with the 'BUILD_HPC' flag.
If you're unit testing, or using one of these logic functions: isfinite(), isinf(), isnan(), isposinf(), isneginf(); then we recommend you to disable the HPC flag: '-D BUILD_HPC=OFF' to obtain the expected results.")
endif()