#cmake_minimum_required(VERSION 3.9.2)
include(ExternalProject)

option(USE_SYSTEM_EIGEN "Use system dependency" OFF)
option(USE_SYSTEM_GTEST "Use system dependency" OFF)
option(USE_SYSTEM_ZLIB "Use system dependency" OFF)
option(USE_SYSTEM_PROTOBUF "Use system dependency" OFF)

# Set variables
set(EDDL_DEPENDENCIES )

#############
##  EIGEN  ##
#############
message(STATUS "Subproject: EIGEN...")
if(USE_SYSTEM_EIGEN)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
else()
    include(eigen.cmake)
endif()
set(EDDL_DEPENDENCIES ${EDDL_DEPENDENCIES} Eigen3)
message(STATUS "Subproject: EIGEN...DONE")


###############
####  GTEST  ##
###############
message(STATUS "Subproject: GTEST...")
if(USE_SYSTEM_GTEST)
    find_package(GTest REQUIRED)
else()
    include(googletest.cmake)
endif()
set(EDDL_DEPENDENCIES ${EDDL_DEPENDENCIES} GTest)
message(STATUS "Subproject: GTEST...DONE")


############
##  ZLIB  ##
############
message(STATUS "Subproject: ZLIB...")
if(USE_SYSTEM_ZLIB)
    find_package(ZLIB REQUIRED)
else()
    include(zlib.cmake)
endif()
set(EDDL_DEPENDENCIES ${EDDL_DEPENDENCIES} ZLIB)
message(STATUS "Subproject: ZLIB...DONE")


################
##  PROTOBUF  ##
################
if(USE_SYSTEM_PROTOBUF)
    find_package(Protobuf REQUIRED)
else()
    include(protobuf.cmake)
endif()
set(EDDL_DEPENDENCIES ${EDDL_DEPENDENCIES} protobuf)
message(STATUS "Subproject: Protobuf...DONE")


message("=== SUPERBUILD ======================>")
message("Eigen3_DIR: ${Eigen3_DIR}")
message("GTEST_ROOT: ${GTEST_ROOT}")
message("ZLIB_ROOT: ${ZLIB_ROOT}")
message("Protobuf_DIR: ${Protobuf_DIR}")
message("USE_SUPERBUILD: ${USE_SUPERBUILD}")


#############
###  EDDL  ##
#############
ExternalProject_Add(ep_eddl
        PREFIX ep_eddl
        SOURCE_DIR "${PROJECT_SOURCE_DIR}"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/ep_eddl"
        INSTALL_COMMAND ""
        CMAKE_CACHE_ARGS
            -DUSE_SUPERBUILD:BOOL=OFF

            -DEigen3_DIR:PATH=${Eigen3_DIR}

            -DGTEST_ROOT:PATH=${GTEST_ROOT}

            #-DZLIB_ROOT:PATH=${ZLIB_ROOT}  # Problems
            -DZLIB_INCLUDE_DIRS:PATH=${ZLIB_INCLUDE_DIRS}
            -DZLIB_LIBRARIES:PATH=${ZLIB_LIBRARIES}
            -DZLIB_LIBRARIES_DIRS:PATH=${ZLIB_LIBRARIES_DIRS}

            #-DProtobuf_DIR:PATH=${Protobuf_DIR}  # Problems
            -DProtobuf_INCLUDE_DIRS:PATH=${Protobuf_INCLUDE_DIRS}
            -DProtobuf_LIBRARIES:PATH=${Protobuf_LIBRARIES}
            -DProtobuf_LIBRARIES_DIRS:PATH=${Protobuf_LIBRARIES_DIRS}
            -DProtobuf_PROTOC_EXECUTABLE:PATH=${Protobuf_PROTOC_EXECUTABLE}

        DEPENDS ${EDDL_DEPENDENCIES}
        )