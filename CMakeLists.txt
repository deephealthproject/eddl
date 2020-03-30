cmake_minimum_required(VERSION 3.9.2)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

PROJECT(eddl VERSION 0.4.5 LANGUAGES CXX)

## Apple: Don't modify install_name when touching RPATH.
if (APPLE)
    SET(CMAKE_MACOSX_RPATH 1)
endif()


# Force default build type to Release
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")

# Use find_package everywhere, no-op if it's a target
macro(find_package)
    if(NOT TARGET ${ARGV0} AND NOT TARGET ${ARGV0}::${ARGV0})
        _find_package(${ARGV})
    else()
        if(TARGET ${ARGV0}::${ARGV0})
            get_target_property(TGT_VER ${ARGV0}::${ARGV0} VERSION)
            set(TGT ${ARGV0}::${ARGV0})
        else()
            get_target_property(TGT_VER ${ARGV0} VERSION)
            set(TGT ${ARGV0})
        endif()
        message(STATUS "Found ${ARGV0}: CMake Target ${TGT} (found version \"${TGT_VER}\")")
        set(${ARGV0}_FOUND TRUE)
    endif()
endmacro()

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


# Install
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
#install(FILES
#        "${PROJECT_BINARY_DIR}/CMakeExternalConfig.cmake"
#        "${PROJECT_BINARY_DIR}/CMakeExternalConfigVersion.cmake"
#        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CMakeExternal"
#        COMPONENT Devel)