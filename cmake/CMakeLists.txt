#cmake_minimum_required(VERSION 3.9.2)
#
#
##############
###  BOOST  ##
##############
##message(STATUS "Subproject: BOOST...")
### Download and unpack eigen at configure time
##configure_file(
##        ${CMAKE_CURRENT_SOURCE_DIR}/boost.CMakeLists.txt
##        ${CMAKE_BINARY_DIR}/boost-download/CMakeLists.txt)
##execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
##        RESULT_VARIABLE result
##        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/boost-download)
##if(result)
##    message(FATAL_ERROR "CMake step for boost failed: ${result}")
##endif()
##execute_process(COMMAND ${CMAKE_COMMAND} --build .
##        RESULT_VARIABLE result
##        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/boost-download)
##if(result)
##    message(FATAL_ERROR "Build step for boost failed: ${result}")
##endif()
##
##add_subdirectory(
##        ${CMAKE_BINARY_DIR}/boost-src
##        ${CMAKE_BINARY_DIR}/boost-build)
##message(STATUS "Subproject: BOOST...DONE")
#
#
###############
####  EIGEN  ##
###############
message(STATUS "Subproject: EIGEN...")
# Download and unpack eigen at configure time
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/eigen.CMakeLists.txt
        ${CMAKE_CURRENT_BINARY_DIR}/eigen-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen-download)
if(result)
    message(FATAL_ERROR "CMake step for eigen failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen-download)
if(result)
    message(FATAL_ERROR "Build step for eigen failed: ${result}")
endif()

message(STATUS "Subproject: EIGEN...DONE")


##############
###  GTEST  ##
##############
message(STATUS "Subproject: GTEST...")
# Download and unpack gtest at configure time
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/gtest.CMakeLists.txt
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-src
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif()

message(STATUS "Subproject: GTEST...DONE")


#################
###  PROTOBUF  ##
#################
message(STATUS "Subproject: Protobuf...")
# Download and unpack gtest at configure time
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/protobuf.CMakeLists.txt
        ${CMAKE_CURRENT_BINARY_DIR}/protobuf-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/protobuf-download )
if(result)
    message(FATAL_ERROR "CMake step for protobuf failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/protobuf-download )
if(result)
    message(FATAL_ERROR "Build step for protobuf failed: ${result}")
endif()

# Protobuf flags
set(protobuf_BUILD_TESTS OFF CACHE BOOL "")
#set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(protobuf_WITH_ZLIB OFF CACHE BOOL "")

# Add protobuf directly to our build. This defines
add_subdirectory(
        ${CMAKE_CURRENT_BINARY_DIR}/protobuf-src/cmake
        ${CMAKE_CURRENT_BINARY_DIR}/protobuf-build
        EXCLUDE_FROM_ALL
        )

message(STATUS "Subproject: Protobuf...DONE")

#############
###  ZLIB  ##
#############
#message(STATUS "Subproject: ZLIB...")
## Download and unpack zlib at configure time
#configure_file(
#        ${CMAKE_CURRENT_SOURCE_DIR}/zlib.CMakeLists.txt
#        ${CMAKE_BINARY_DIR}/zlib-download/CMakeLists.txt)
#execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
#        RESULT_VARIABLE result
#        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/zlib-download)
#if(result)
#    message(FATAL_ERROR "CMake step for zlib failed: ${result}")
#endif()
#execute_process(COMMAND ${CMAKE_COMMAND} --build .
#        RESULT_VARIABLE result
#        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/zlib-download)
#if(result)
#    message(FATAL_ERROR "Build step for zlib failed: ${result}")
#endif()
#
#add_subdirectory(
#        ${CMAKE_BINARY_DIR}/zlib-src
#        ${CMAKE_BINARY_DIR}/zlib-build)
#message(STATUS "Subproject: ZLIB...DONE")