cmake_minimum_required(VERSION 3.9.2)


#############
##  BOOST  ##
#############
#message(STATUS "Subproject: BOOST...")
## Download and unpack eigen at configure time
#configure_file(
#        ${CMAKE_CURRENT_SOURCE_DIR}/boost.CMakeLists.txt
#        ${CMAKE_BINARY_DIR}/boost-download/CMakeLists.txt)
#execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
#        RESULT_VARIABLE result
#        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/boost-download)
#if(result)
#    message(FATAL_ERROR "CMake step for boost failed: ${result}")
#endif()
#execute_process(COMMAND ${CMAKE_COMMAND} --build .
#        RESULT_VARIABLE result
#        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/boost-download)
#if(result)
#    message(FATAL_ERROR "Build step for boost failed: ${result}")
#endif()
#
#add_subdirectory(
#        ${CMAKE_BINARY_DIR}/boost-src
#        ${CMAKE_BINARY_DIR}/boost-build)
#message(STATUS "Subproject: BOOST...DONE")


##############
###  EIGEN  ##
##############
message(STATUS "Subproject: EIGEN...")
# Download and unpack eigen at configure time
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/eigen.CMakeLists.txt
        ${CMAKE_BINARY_DIR}/eigen-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/eigen-download)
if(result)
    message(FATAL_ERROR "CMake step for gtest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/eigen-download)
if(result)
    message(FATAL_ERROR "Build step for eigen failed: ${result}")
endif()

add_subdirectory(
        ${CMAKE_BINARY_DIR}/eigen-src
        ${CMAKE_BINARY_DIR}/eigen-build)
message(STATUS "Subproject: EIGEN...DONE")


#############
##  GTEST  ##
#############
message(STATUS "Subproject: GTEST...")
# Download and unpack gtest at configure time
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/gtest.CMakeLists.txt
        ${CMAKE_BINARY_DIR}/gtest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gtest-download)
if(result)
    message(FATAL_ERROR "CMake step for gtest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gtest-download)
if(result)
    message(FATAL_ERROR "Build step for gtest failed: ${result}")
endif()

add_subdirectory(
        ${CMAKE_BINARY_DIR}/gtest-src
        ${CMAKE_BINARY_DIR}/gtest-build)
message(STATUS "Subproject: GTEST...DONE")


################
##  PROTOBUF  ##
################
message(STATUS "Subproject: Protobuf...")
# Download and unpack protobuf at configure time
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/protobuf.CMakeLists.txt
        ${CMAKE_BINARY_DIR}/protobuf-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/protobuf-download )
if(result)
    message(FATAL_ERROR "CMake step for protobuf failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/protobuf-download )
if(result)
    message(FATAL_ERROR "Build step for protobuf failed: ${result}")
endif()

set(protobuf_BUILD_TESTS OFF CACHE INTERNAL "Disable Protobuf tests")
add_subdirectory(
        ${CMAKE_BINARY_DIR}/protobuf-src/cmake
        ${CMAKE_BINARY_DIR}/protobuf-build)
message(STATUS "Subproject: Protobuf...DONE")


############
##  ZLIB  ##
############
message(STATUS "Subproject: ZLIB...")
# Download and unpack zlib at configure time
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/zlib.CMakeLists.txt
        ${CMAKE_BINARY_DIR}/zlib-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/zlib-download)
if(result)
    message(FATAL_ERROR "CMake step for zlib failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/zlib-download)
if(result)
    message(FATAL_ERROR "Build step for zlib failed: ${result}")
endif()

add_subdirectory(
        ${CMAKE_BINARY_DIR}/zlib-src
        ${CMAKE_BINARY_DIR}/zlib-build)
message(STATUS "Subproject: ZLIB...DONE")