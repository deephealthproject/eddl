#cmake_minimum_required(VERSION 3.9.2)
include(ExternalProject)

# Set parallelization
include(ProcessorCount)
ProcessorCount(NUM_CORES)
SET(NUM_PARALLEL 1)
if (NUM_CORES GREATER_EQUAL 2)
    math(EXPR NUM_PARALLEL "${NUM_CORES} - 1")
endif()

# For development (ignore)
option(USE_SYSTEM_DEPENDENCIES "Force the use of system dependencies" OFF)  # Master flag
option(USE_SYSTEM_GTEST "Use system dependency" OFF)
option(USE_SYSTEM_EIGEN "Use system dependency" OFF)
option(USE_SYSTEM_ZLIB "Use system dependency" OFF)
option(USE_SYSTEM_PROTOBUF "Use system dependency" OFF)

# Set variables
set(EDDL_DEPENDENCIES)

#################################################################################
#################################################################################

##############
###  GTEST  ##
##############
message(STATUS "Subproject: GTEST...")
if(USE_SYSTEM_DEPENDENCIES OR USE_SYSTEM_GTEST)
    find_package(GTest REQUIRED)
else()

    # Download and unpack googletest at configure time
    configure_file(googletest.CMakeLists.txt.in googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build . -j ${NUM_PARALLEL}
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    # Set variables
    SET(GTEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}/googletest")
endif()
message(STATUS "Subproject: GTEST...DONE")


#############
##  EIGEN  ##
#############
message(STATUS "Subproject: EIGEN...")
if(USE_SYSTEM_DEPENDENCIES OR USE_SYSTEM_EIGEN)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
else()

    # Download and unpack googletest at configure time
    configure_file(eigen.CMakeLists.txt.in eigen-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen-download )
    if(result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build . -j ${NUM_PARALLEL}
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen-download )
    if(result)
        message(FATAL_ERROR "Build step for eigen failed: ${result}")
    endif()

    # Set variables
    SET(Eigen3_DIR "${CMAKE_CURRENT_BINARY_DIR}/eigen/share/eigen3/cmake" )
#        find_package(Eigen3 3.3 REQUIRED NO_MODULE)  # Double check - Fill more variables
endif()
message(STATUS "Subproject: EIGEN...DONE")


#############
###  ZLIB  ##
#############
message(STATUS "Subproject: ZLIB...")
if(USE_SYSTEM_DEPENDENCIES OR USE_SYSTEM_ZLIB)
    find_package(ZLIB REQUIRED)
else()
    # Download and unpack googletest at configure time
    configure_file(zlib.CMakeLists.txt.in zlib-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/zlib-download )
    if(result)
        message(FATAL_ERROR "CMake step for zlib failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build . -j ${NUM_PARALLEL}
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/zlib-download )
    if(result)
        message(FATAL_ERROR "Build step for zlib failed: ${result}")
    endif()


    # Set variables
    SET(ZLIB_ROOT "${CMAKE_CURRENT_BINARY_DIR}/zlib")
    SET(ZLIB_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/zlib/include")
    SET(ZLIB_LIBRARIES_DIRS "${CMAKE_CURRENT_BINARY_DIR}/zlib/lib")
    SET(ZLIB_LIBRARIES "z")

#    find_library(ZLIB_LIBRARIES
#            NAMES z
#            PATHS ${ZLIB_LIBRARIES_DIRS}
#            PATH_SUFFIXES lib
#            NO_DEFAULT_PATH
#            )
#    if(NOT ZLIB_LIBRARIES)
#        message(FATAL_ERROR "ZLIB library not found")
#    endif()
endif()
message(STATUS "Subproject: ZLIB...DONE")


################
##  PROTOBUF  ##
################
if(USE_SYSTEM_DEPENDENCIES OR USE_SYSTEM_PROTOBUF)
    find_package(Protobuf REQUIRED)
else()
    # Download and unpack googletest at configure time
    configure_file(protobuf.CMakeLists.txt.in protobuf-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/protobuf-download )
    if(result)
        message(FATAL_ERROR "CMake step for protobuf failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build . -j ${NUM_PARALLEL}
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/protobuf-download )
    if(result)
        message(FATAL_ERROR "Build step for protobuf failed: ${result}")
    endif()

    # This variable does not work (deprecated?)
    if (WIN32)
        set(Protobuf_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf/cmake")
    else()
        set(Protobuf_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf/lib/cmake/protobuf")
    endif()

    set(Protobuf_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/protobuf/include")
    set(Protobuf_LIBRARIES_DIRS "${CMAKE_CURRENT_BINARY_DIR}/protobuf/lib")
    set(Protobuf_LIBRARIES "protobuf")
    set(Protobuf_PROTOC_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/protobuf/bin/protoc")

#        # Weirdly, in github does not work
#        find_library(Protobuf_LIBRARIES
#                NAMES protobuf protobuf-lite protoc
#                PATHS ${Protobuf_LIBRARIES_DIRS}
#                PATH_SUFFIXES lib
#                NO_DEFAULT_PATH
#                )
#        if(NOT Protobuf_LIBRARIES)
#            message(FATAL_ERROR "Protobuf library not found")
#        endif()

endif()
message(STATUS "Subproject: Protobuf...DONE")

##############
####  EDDL  ##
##############
ExternalProject_Add(ep_eddl
        PREFIX ep_eddl
        SOURCE_DIR "${PROJECT_SOURCE_DIR}"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/ep_eddl"
        INSTALL_COMMAND ""
        CMAKE_CACHE_ARGS
            -DUSE_SUPERBUILD:BOOL=OFF

            -DEigen3_DIR:PATH=${Eigen3_DIR}

            -DGTEST_ROOT:PATH=${GTEST_ROOT}  # Special

            -DZLIB_ROOT:PATH=${ZLIB_ROOT}  # Problems
            -DZLIB_INCLUDE_DIRS:PATH=${ZLIB_INCLUDE_DIRS}
            -DZLIB_LIBRARIES:PATH=${ZLIB_LIBRARIES}
            -DZLIB_LIBRARIES_DIRS:PATH=${ZLIB_LIBRARIES_DIRS}

            -DProtobuf_DIR:PATH=${Protobuf_DIR}  # Problems
            -DProtobuf_INCLUDE_DIRS:PATH=${Protobuf_INCLUDE_DIRS}
            -DProtobuf_LIBRARIES:PATH=${Protobuf_LIBRARIES}
            -DProtobuf_LIBRARIES_DIRS:PATH=${Protobuf_LIBRARIES_DIRS}
            -DProtobuf_PROTOC_EXECUTABLE:PATH=${Protobuf_PROTOC_EXECUTABLE}

        DEPENDS ${EDDL_DEPENDENCIES}
        )


# SHOW OPTIONS
message(STATUS "===========================================" )
message(STATUS "===========================================" )
message(STATUS "Use system dependencies: " ${USE_SYSTEM_DEPENDENCIES} )
message(STATUS "Cores used: " ${NUM_PARALLEL} )
message(STATUS "-------------------------------------------" )
message(STATUS "Use system GTest: " ${USE_SYSTEM_GTEST} )
message(STATUS "GTests root: " ${GTEST_ROOT})
message(STATUS "-------------------------------------------" )
message(STATUS "Use system Eigen: " ${USE_SYSTEM_EIGEN} )
message(STATUS "Eigen dir: " ${Eigen3_DIR})
message(STATUS "-------------------------------------------" )
message(STATUS "Use system Zlib: " ${USE_SYSTEM_ZLIB} )
message(STATUS "Zlib root: " ${ZLIB_ROOT})
message(STATUS "Zlib include dirs: " ${ZLIB_INCLUDE_DIRS})
message(STATUS "Zlib libraries dirs: " ${ZLIB_LIBRARIES_DIRS})
message(STATUS "Zlib libraries: " ${ZLIB_LIBRARIES})
message(STATUS "-------------------------------------------" )
message(STATUS "Use system Protobuf: " ${USE_SYSTEM_PROTOBUF} )
message(STATUS "Protobuf dir: " ${Protobuf_DIR})
message(STATUS "Protobuf include dirs: " ${Protobuf_INCLUDE_DIRS})
message(STATUS "Protobuf libraries dirs: " ${Protobuf_LIBRARIES_DIRS})
message(STATUS "Protobuf libraries: " ${Protobuf_LIBRARIES})
message(STATUS "Protobuf compiler: " ${Protobuf_PROTOC_EXECUTABLE})
message(STATUS "===========================================" )
message(STATUS "===========================================" )
