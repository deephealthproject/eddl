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

if(NOT BUILD_PROTOBUF)
    list(FILTER CPP_TESTS_FILES EXCLUDE REGEX ".*/onnx/*")
endif()

# Build test and target libraries
add_executable(unit_tests ${CPP_TESTS_FILES})
target_include_directories(unit_tests PUBLIC $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>)  # TODO: Why build interface?


if(MSVC)
    target_link_libraries(unit_tests PUBLIC eddl 
        optimized ${GTEST_LIBRARY} optimized ${GTEST_MAIN_LIBRARY}
        debug ${GTESTD_LIBRARY} debug ${GTEST_MAIND_LIBRARY}
    )
else()
    find_package(Threads)
    target_link_libraries(unit_tests PUBLIC eddl ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
endif()


# CUDA (set in parent scope)
if(USE_CUDA)
    add_definitions(-DcGPU)
endif()
if(USE_CUDNN)
    add_definitions(-DcCUDNN)
endif()
if (USE_FPGA)
    add_definitions(-DcFPGA)
endif()

# Add test
add_test(NAME unit_tests COMMAND unit_tests)


# SHOW OPTIONS
message(STATUS "===========================================" )
message(STATUS "===========================================" )
message(STATUS "GTest root: " ${GTEST_ROOT} )
message(STATUS "GTest include dirs: " ${GTEST_INCLUDE_DIRS} )
message(STATUS "GTest libraries: ${GTEST_LIBRARY}; ${GTEST_MAIN_LIBRARY} ")
#message(STATUS "GTest libraries (debug): ${GTESTD_LIBRARY}; ${GTESTD_MAIN_LIBRARY} ")
message(STATUS "===========================================" )
message(STATUS "===========================================" )
