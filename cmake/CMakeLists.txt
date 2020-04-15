#cmake_minimum_required(VERSION 3.9.2)
include(ExternalProject)

set_property (DIRECTORY PROPERTY EP_BASE third_party)
SET(EP_BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party")

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

if(USE_SYSTEM_DEPENDENCIES)
    set(USE_SYSTEM_GTEST OFF)
    set(USE_SYSTEM_EIGEN OFF)
    set(USE_SYSTEM_ZLIB OFF)
    set(USE_SYSTEM_PROTOBUF OFF)
endif()

# Collect args
get_cmake_property(vars CACHE_VARIABLES)
foreach(var ${vars})
    get_property(currentHelpString CACHE "${var}" PROPERTY HELPSTRING)
    if("${currentHelpString}" MATCHES "No help, variable specified on the command line." OR "${currentHelpString}" STREQUAL "")
        # message("${var} = [${${var}}]  --  ${currentHelpString}") # uncomment to see the variables being processed
        list(APPEND CL_ARGS "-D${var}=${${var}}")
    endif()
endforeach()

#################################################################################
#################################################################################

##############
###  GTEST  ##
##############
message(STATUS "Subproject: GTEST...")
if(USE_SYSTEM_GTEST)
    find_package(GTest REQUIRED)
else()

    # Download and unpack googletest at configure time
    configure_file(googletest.CMakeLists.txt.in ${EP_BASE_DIR}/googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${EP_BASE_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build . -j ${NUM_PARALLEL}
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${EP_BASE_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    # Set variables
    SET(GTEST_ROOT "${EP_BASE_DIR}/googletest" PARENT_SCOPE)
endif()
message(STATUS "Subproject: GTEST...DONE")

#
##############
###  EIGEN  ##
##############
#message(STATUS "Subproject: EIGEN...")
#if(USE_SYSTEM_EIGEN)
#    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#else()
#
#    # Download and unpack googletest at configure time
#    configure_file(eigen.CMakeLists.txt.in ${EP_BASE_DIR}/eigen-download/CMakeLists.txt)
#    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
#            RESULT_VARIABLE result
#            WORKING_DIRECTORY ${EP_BASE_DIR}/eigen-download )
#    if(result)
#        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
#    endif()
#    execute_process(COMMAND ${CMAKE_COMMAND} --build . -j ${NUM_PARALLEL}
#            RESULT_VARIABLE result
#            WORKING_DIRECTORY ${EP_BASE_DIR}/eigen-download )
#    if(result)
#        message(FATAL_ERROR "Build step for eigen failed: ${result}")
#    endif()
#
#    # Set variables
#    SET(Eigen3_DIR "${EP_BASE_DIR}/eigen/share/eigen3/cmake" )
##        find_package(Eigen3 3.3 REQUIRED NO_MODULE)  # Double check - Fill more variables
#endif()
#message(STATUS "Subproject: EIGEN...DONE")


#############
###  ZLIB  ##
#############
message(STATUS "Subproject: ZLIB...")
if(USE_SYSTEM_ZLIB)
    find_package(ZLIB REQUIRED)
else()
    # Download and unpack googletest at configure time
    configure_file(zlib.CMakeLists.txt.in ${EP_BASE_DIR}/zlib-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${EP_BASE_DIR}/zlib-download )
    if(result)
        message(FATAL_ERROR "CMake step for zlib failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build . -j ${NUM_PARALLEL}
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${EP_BASE_DIR}/zlib-download )
    if(result)
        message(FATAL_ERROR "Build step for zlib failed: ${result}")
    endif()

    # Set variables
    SET(ZLIB_ROOT "${EP_BASE_DIR}/zlib" PARENT_SCOPE)
    SET(ZLIB_INCLUDE_DIRS "${EP_BASE_DIR}/zlib/include" PARENT_SCOPE)
    SET(ZLIB_LIBRARIES_DIRS "${EP_BASE_DIR}/zlib/lib" PARENT_SCOPE)
    SET(ZLIB_LIBRARIES "z" PARENT_SCOPE)
endif()
message(STATUS "Subproject: ZLIB...DONE")


################
##  PROTOBUF  ##
################
if(USE_SYSTEM_PROTOBUF)
    find_package(Protobuf REQUIRED)
else()
    # Download and unpack googletest at configure time
    configure_file(protobuf.CMakeLists.txt.in ${EP_BASE_DIR}/protobuf-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${EP_BASE_DIR}/protobuf-download )
    if(result)
        message(FATAL_ERROR "CMake step for protobuf failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build . -j ${NUM_PARALLEL}
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${EP_BASE_DIR}/protobuf-download )
    if(result)
        message(FATAL_ERROR "Build step for protobuf failed: ${result}")
    endif()

    # This variable does not work (deprecated?)
    if (WIN32)
        set(Protobuf_DIR "${EP_BASE_DIR}/protobuf/cmake" PARENT_SCOPE)
    else()
        set(Protobuf_DIR "${EP_BASE_DIR}/protobuf/lib/cmake/protobuf" PARENT_SCOPE)
    endif()

    set(Protobuf_INCLUDE_DIRS "${EP_BASE_DIR}/protobuf/include" PARENT_SCOPE)
    set(Protobuf_LIBRARIES_DIRS "${EP_BASE_DIR}/protobuf/lib" PARENT_SCOPE)
    set(Protobuf_LIBRARIES "protobuf" PARENT_SCOPE)
    set(Protobuf_PROTOC_EXECUTABLE "${EP_BASE_DIR}/protobuf/bin/protoc" PARENT_SCOPE)
endif()
message(STATUS "Subproject: Protobuf...DONE")
#
###############
#####  EDDL  ##
###############
#ExternalProject_Add(eddl
#        PREFIX eddl
#        DEPENDS ${EDDL_DEPENDENCIES}
#        SOURCE_DIR "${PROJECT_SOURCE_DIR}"
#        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/eddl"
#        INSTALL_COMMAND ""
#        CMAKE_ARGS ${CL_ARGS}  # Overrides CMAKE_CACHE_ARGS when an argument is duplicated
#        CMAKE_CACHE_ARGS
#            -DUSE_SUPERBUILD:BOOL=OFF
#            -DUSE_SYSTEM_EIGEN:BOOL=${USE_SYSTEM_EIGEN}
#            -DUSE_SYSTEM_GTEST:BOOL=${USE_SYSTEM_GTEST}
#            -DUSE_SYSTEM_ZLIB:BOOL=${USE_SYSTEM_ZLIB}
#            -DUSE_SYSTEM_PROTOBUF:BOOL=${USE_SYSTEM_PROTOBUF}
#
##            -DEigen3_DIR:PATH=${Eigen3_DIR}
#
#            -DGTEST_ROOT:PATH=${GTEST_ROOT}  # Special
#
#            -DZLIB_ROOT:PATH=${ZLIB_ROOT}  # Problems
#            -DZLIB_INCLUDE_DIRS:PATH=${ZLIB_INCLUDE_DIRS}
#            -DZLIB_LIBRARIES:PATH=${ZLIB_LIBRARIES}
#            -DZLIB_LIBRARIES_DIRS:PATH=${ZLIB_LIBRARIES_DIRS}
#
#            -DProtobuf_DIR:PATH=${Protobuf_DIR}  # Problems
#            -DProtobuf_INCLUDE_DIRS:PATH=${Protobuf_INCLUDE_DIRS}
#            -DProtobuf_LIBRARIES:PATH=${Protobuf_LIBRARIES}
#            -DProtobuf_LIBRARIES_DIRS:PATH=${Protobuf_LIBRARIES_DIRS}
#            -DProtobuf_PROTOC_EXECUTABLE:PATH=${Protobuf_PROTOC_EXECUTABLE}
#        )


# SHOW OPTIONS
message(STATUS "===========================================" )
message(STATUS "===========================================" )
message(STATUS "Use system dependencies: " ${USE_SYSTEM_DEPENDENCIES} )
message(STATUS "Cores used: " ${NUM_PARALLEL} )
message(STATUS "Args used: " ${CL_ARGS} )
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
