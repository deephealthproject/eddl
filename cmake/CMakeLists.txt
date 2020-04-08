#cmake_minimum_required(VERSION 3.9.2)
include(ExternalProject)


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
    include(googletest.cmake)
    set(EDDL_DEPENDENCIES ${EDDL_DEPENDENCIES} GTest)
endif()
message(STATUS "Subproject: GTEST...DONE")

#############
##  EIGEN  ##
#############
message(STATUS "Subproject: EIGEN...")
if(USE_SYSTEM_DEPENDENCIES OR USE_SYSTEM_EIGEN)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
else()
    include(eigen.cmake)
    set(EDDL_DEPENDENCIES ${EDDL_DEPENDENCIES} Eigen3)
endif()
message(STATUS "Subproject: EIGEN...DONE")


############
##  ZLIB  ##
############
message(STATUS "Subproject: ZLIB...")
if(USE_SYSTEM_DEPENDENCIES OR USE_SYSTEM_ZLIB)
    find_package(ZLIB REQUIRED)
else()
    include(zlib.cmake)
    set(EDDL_DEPENDENCIES ${EDDL_DEPENDENCIES} ZLIB)
endif()
message(STATUS "Subproject: ZLIB...DONE")


################
##  PROTOBUF  ##
################
if(USE_SYSTEM_DEPENDENCIES OR USE_SYSTEM_PROTOBUF)
    find_package(Protobuf REQUIRED)
else()
    include(protobuf.cmake)
    set(EDDL_DEPENDENCIES ${EDDL_DEPENDENCIES} protobuf)
endif()
message(STATUS "Subproject: Protobuf...DONE")

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
message(STATUS "Zlib libraries: " ${ZLIB_LIBRARIES})
message(STATUS "-------------------------------------------" )
message(STATUS "Use system Protobuf: " ${USE_SYSTEM_PROTOBUF} )
message(STATUS "Protobuf dir: " ${Protobuf_DIR})
message(STATUS "Protobuf include dirs: " ${Protobuf_INCLUDE_DIRS})
message(STATUS "Protobuf libraries: " ${Protobuf_LIBRARIES})
message(STATUS "Protobuf compiler: " ${Protobuf_PROTOC_EXECUTABLE})
message(STATUS "===========================================" )
message(STATUS "===========================================" )
