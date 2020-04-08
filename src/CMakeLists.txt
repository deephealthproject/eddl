###########################################################################
################################ OPTIONS ##################################
###########################################################################

# USER OPTIONS *******************************
set(BUILD_TARGET "GPU" CACHE STRING "Compile library for {CPU, GPU, FPGA}")
option(BUILD_OPENMP "Compile using OpenMP" ON)
option(BUILD_PROTOBUF "Compile using Protobuf" ON)
option(OVERWRITE_PROTO_FILES "Overwrite protobuf files (requires a compatible protobuf compiler)" ON)

# Double checks (restricted args)
set_property(CACHE BUILD_TARGET PROPERTY STRINGS CPU GPU FPGA)

# Device specific setup
string(TOUPPER ${BUILD_TARGET} BUILD_TARGET)  # Detect cpu, Cpu, CPU,...
if(${BUILD_TARGET} STREQUAL "CPU")
elseif(${BUILD_TARGET} STREQUAL "GPU")
    set(USE_CUDA ON CACHE BOOL "Enable CUDA support if checked")
elseif(${BUILD_TARGET} STREQUAL "FPGA")
    message(FATAL_ERROR "FPGA not implemented, CMake will exit." )
else()
    message(FATAL_ERROR "Build target not implemented, CMake will exit." )
endif()

# Force default build type to Release
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")


###########################################################################
############################# SET LIBRARY #################################
###########################################################################

# C++ files
file(GLOB_RECURSE CPP_HEADERS "${PROJECT_SOURCE_DIR}/src/*" *.{h, *.hpp})
file(GLOB_RECURSE CPP_SOURCES "${PROJECT_SOURCE_DIR}/src/*" *.{cc, cpp})
file(GLOB_RECURSE CUDA_SOURCES "${PROJECT_SOURCE_DIR}/src/*" *.{cu})

# Make an automatic library - will be static or dynamic based on user setting
add_library(${PROJECT_NAME} ${CPP_HEADERS} ${CPP_SOURCES})

# We need this directory, and users of our library will need it too
target_include_directories(${PROJECT_NAME} PUBLIC
                           ${PROJECT_SOURCE_DIR}/src)

# All users of this library will need at least C++11
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

# IDEs should put the headers in a nice place
#source_group(TREE "${PROJECT_SOURCE_DIR}/include2" PREFIX ${PROJECT_NAME} FILES ${CPP_PUBLIC_HEADERS})


###########################################################################
########################### LINK LIBRARIES ################################
###########################################################################
# Threads
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# Eigen
if(DEFINED EIGEN3_DIR)
    message("Using specific Eigen")
else()
    message("Searching for Eigen in the system...")
endif()
find_package(Eigen3 3.3 REQUIRED NO_MODULE)  # EIGEN_DIR => ok
message("Eigen found!")
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)  # Header only library

# ZLIB
if(DEFINED ZLIB_INCLUDE_DIRS AND DEFINED ZLIB_LIBRARIES_DIRS AND DEFINED ZLIB_LIBRARIES)
    message("Using specific Zlib")
else()
    message("Searching for ZLIB in the system...")
    find_package(ZLIB REQUIRED)  # Problems with: ZLIB_ROOT
    message("ZLIB found!")
endif()
#target_include_directories(${PROJECT_NAME} PUBLIC ${ZLIB_INCLUDE_DIRS})
#target_link_directories(${PROJECT_NAME} PRIVATE ${ZLIB_LIBRARIES_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${ZLIB_LIBRARIES})

# ONNX files
if(BUILD_PROTOBUF)
    add_definitions(-DcPROTO)

    # Link library
    if(DEFINED Protobuf_INCLUDE_DIRS AND DEFINED Protobuf_LIBRARIES_DIRS AND DEFINED Protobuf_LIBRARIES)
        message("Using specific Protobuf")
    else()
        message("Searching for Protobuf in the system...")
        find_package(Protobuf REQUIRED)  # Problems with: Protobuf_DIR
        message("Protobuf found!")
    endif()
    target_include_directories(${PROJECT_NAME} PUBLIC ${Protobuf_INCLUDE_DIRS})
    target_link_directories(${PROJECT_NAME} PRIVATE ${Protobuf_LIBRARIES_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${Protobuf_LIBRARIES})

    # Create "onnx.pb.cc" and "onnx.pb.h" files (from serialization/onnx)
    # Equivalent to: /usr/local/bin/protoc --cpp_out . onnx.proto
    if(OVERWRITE_PROTO_FILES)
        EXEC_PROGRAM(${Protobuf_PROTOC_EXECUTABLE} ARGS --cpp_out ${PROJECT_SOURCE_DIR}/src/serialization/onnx/ --proto_path ${PROJECT_SOURCE_DIR}/src/serialization/onnx/ onnx.proto )
    endif()
endif()


# OpenMP
if(BUILD_OPENMP)
    find_package(OpenMP)
    if (OpenMP_FOUND)
        target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
        if (${OpenMP_CXX_VERSION_MAJOR})
            set(OpenMP_VERSION_MAJOR ${OpenMP_CXX_VERSION_MAJOR} CACHE INTERNAL "" FORCE)
        endif()
        target_compile_definitions(${PROJECT_NAME} PUBLIC OpenMP_VERSION_MAJOR=${OpenMP_VERSION_MAJOR})
        message(STATUS "Found OpenMP, version ${OpenMP_VERSION_MAJOR}")

        set(OMP_CANCELLATION true)  # https://www.openmp.org/spec-html/5.0/openmpse59.html
    else()
        set(BUILD_OPENMP OFF)
    endif()
endif()


# CUDA
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        add_definitions(-DcGPU)

        target_sources(${PROJECT_NAME} PRIVATE ${CUDA_SOURCES})

        # Target properties
        set_target_properties(${PROJECT_NAME} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
                #CUDA_SEPARABLE_COMPILATION ON
                )

        target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        target_link_libraries(${PROJECT_NAME} PRIVATE cublas cudart curand)

        if(APPLE)
            # We need to add the path to the driver (libcuda.dylib) as an rpath,
            # so that the static cuda runtime can find it at runtime.
            set_property(TARGET ${PROJECT_NAME} PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        endif()

        # TODO: Is this really needed?
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__")
        endif()

    else()
        set(USE_CUDA OFF)
        set(BUILD_TARGET "CPU")
    endif()
endif()


# SHOW OPTIONS
message(STATUS "===========================================" )
message(STATUS "===========================================" )
message(STATUS "Project name: " ${CMAKE_PROJECT_NAME} )
message(STATUS "Project version: " ${CMAKE_PROJECT_VERSION} )
message(STATUS "Prefix path: " ${CMAKE_PREFIX_PATH} )
message(STATUS "Install prefix: " ${CMAKE_INSTALL_PREFIX} )
message(STATUS "-------------------------------------------" )
message(STATUS "Build target: " ${BUILD_TARGET} )
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE} )
message(STATUS "Build shared libs: " ${BUILD_SHARED_LIBS} )
message(STATUS "-------------------------------------------" )
message(STATUS "Use OpenMP: " ${BUILD_OPENMP} )
message(STATUS "Use Protobuf: " ${BUILD_PROTOBUF} )
message(STATUS "Use HPC: " ${BUILD_HPC} )
message(STATUS "Use Intel-MKL: " ${MKL} )
message(STATUS "-------------------------------------------" )
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "C++ flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "C++ flags (release): " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "C++ flags (debug): " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "-------------------------------------------" )
message(STATUS "CUDA Enabled: " ${USE_CUDA} )
message(STATUS "CUDA compiler: " ${CMAKE_CUDA_COMPILER})
message(STATUS "CUDA flags: " ${CMAKE_CUDA_FLAGS} )
message(STATUS "CUDA toolkit: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )
message(STATUS "-------------------------------------------" )
message(STATUS "OpenMP version: " ${OpenMP_VERSION_MAJOR} )
message(STATUS "OpenMP gomp library: " ${OpenMP_gomp_LIBRARY} )
message(STATUS "OpenMP pthread library: " ${OpenMP_pthread_LIBRARY} )
message(STATUS "-------------------------------------------" )
message(STATUS "Eigen3 dir: " ${Eigen3_DIR} )
message(STATUS "Eigen3 include: " ${EIGEN3_INCLUDE_DIR} )
message(STATUS "-------------------------------------------" )
message(STATUS "ZLIB root: " ${ZLIB_ROOT} )
message(STATUS "ZLIB include: " ${ZLIB_INCLUDE_DIRS} )
message(STATUS "ZLIB libraries: " ${ZLIB_LIBRARIES} )
message(STATUS "-------------------------------------------" )
message(STATUS "Protobuf dir: " ${Protobuf_DIR} )
message(STATUS "Protobuf include: " ${Protobuf_INCLUDE_DIRS} )
message(STATUS "Protobuf libraries: " ${Protobuf_LIBRARIES} )
message(STATUS "Protobuf compiler: " ${Protobuf_PROTOC_EXECUTABLE} )
message(STATUS "===========================================" )
message(STATUS "===========================================" )
