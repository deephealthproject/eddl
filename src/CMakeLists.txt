##########################################################################
################################ OPTIONS ##################################
###########################################################################

# USER OPTIONS *******************************
#CUDNN option is CUDA+CUDNN
set(BUILD_TARGET "CUDNN" CACHE STRING "Compile library for {CPU, GPU, CUDNN, FPGA}")
#option(OVERWRITE_PROTO_FILES "Overwrite Protobuf files (requires a compatible Protobuf compiler)" ON)

# Double checks (restricted args)
set_property(CACHE BUILD_TARGET PROPERTY STRINGS CPU GPU CUDNN FPGA)

# Initializations (Local's scope)
SET(USE_OPENMP OFF)
SET(USE_CUDA OFF)
SET(USE_CUDNN OFF)
SET(USE_FPGA OFF)

# Device specific setup
string(TOUPPER ${BUILD_TARGET} BUILD_TARGET)  # Detect cpu, Cpu, CPU,...
if(${BUILD_TARGET} STREQUAL "CPU")
elseif(${BUILD_TARGET} STREQUAL "GPU")
    SET(USE_CUDA ON) # Local's scope
elseif(${BUILD_TARGET} STREQUAL "CUDNN")
    SET(USE_CUDA ON) # Local's scope
    SET(USE_CUDNN ON) # Local's scope
elseif(${BUILD_TARGET} STREQUAL "FPGA")
    SET(USE_FPGA ON) # Local's scope
else()
    message(FATAL_ERROR "Build target '${BUILD_TARGET}' not implemented, CMake will exit." )
endif()

# Force default build type to Release
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")


###########################################################################
############################### SANITY CHECKS #############################
###########################################################################

# Prefer static libraries
# There are problems since not all static libraries have been compile with -fPIC
#IF(WIN32)
#    SET(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
#ELSE(WIN32)
#    SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
#ENDIF(WIN32)

###########################################################################
############################### GET FILES #################################
###########################################################################

# C++ files
file(GLOB_RECURSE PUBLIC_HEADERS "${PROJECT_SOURCE_DIR}/include/*" *.{h, hpp})
file(GLOB_RECURSE CPP_HEADERS "${PROJECT_SOURCE_DIR}/src/*" *.{h, hpp})
file(GLOB_RECURSE CPP_SOURCES "${PROJECT_SOURCE_DIR}/src/*" *.{cc, cpp})

# Remove dependent files of protobuf
list(FILTER PUBLIC_HEADERS EXCLUDE REGEX ".*/include/eddl/serialization/onnx/onnx.pb*")
list(FILTER CPP_HEADERS EXCLUDE REGEX ".*/src/serialization/onnx/onnx_third_party/onnx.pb*")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*/src/serialization/onnx/onnx_third_party/onnx.pb*")

# Remove problematic files if they are not needed
if(NOT BUILD_DIST)
    list(FILTER PUBLIC_HEADERS EXCLUDE REGEX ".*/src/distributed/*")
    list(FILTER CPP_HEADERS EXCLUDE REGEX ".*/src/distributed/*")
    list(FILTER CPP_SOURCES EXCLUDE REGEX ".*/src/distributed/*")
endif()

SET(ALL_FILES ${PUBLIC_HEADERS} ${CPP_HEADERS} ${CPP_SOURCES})

# Setup-dependent  files (dependent on the setup)  ########################
# ONNX (these files might not exist yet)
set(ONNX_FILES ${PROJECT_SOURCE_DIR}/include/eddl/serialization/onnx/onnx.pb.h
        ${PROJECT_SOURCE_DIR}/src/serialization/onnx/onnx_third_party/onnx.pb.cc)

# CUDA & CUDNN
file(GLOB_RECURSE CUDA_HEADERS "${PROJECT_SOURCE_DIR}/include/eddl/hardware/gpu/*" *.{h, hpp})
file(GLOB_RECURSE CUDA_SOURCES "${PROJECT_SOURCE_DIR}/src/hardware/gpu/*" *.{cu})

# FPGA
file(GLOB_RECURSE FPGA_HEADERS "${PROJECT_SOURCE_DIR}/include/eddl/hardware/fpga/*" *.{h, hpp})
file(GLOB_RECURSE FPGA_SOURCES "${PROJECT_SOURCE_DIR}/src/hardware/fpga/*" *.{cc, cpp})

# Remove setup-dependent files (later we'll add the ones we need)
SET(SPECIAL_FILES ${CUDA_HEADERS} ${CUDA_SOURCES} ${FPGA_HEADERS} ${FPGA_SOURCES})
FOREACH(item ${SPECIAL_FILES})
    LIST(REMOVE_ITEM ALL_FILES ${item})
ENDFOREACH(item)



###########################################################################
############################# SET LIBRARY #################################
###########################################################################

# Make an automatic library - will be static or dynamic based on user setting
add_library(${PROJECT_NAME} ${ALL_FILES})

# We need this directory, and users of our library will need it too
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        )

# All users of this library will need at least C++11
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

# IDEs should put the headers in a nice place
#source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${CPP_HEADERS})

if(MSVC)
    # Install the debug library with "d" suffix
    set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")
endif()

###########################################################################
########################### LINK LIBRARIES ################################
###########################################################################

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)  # EIGEN_DIR => ok
target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)  # Header only library

# ONNX files
if(BUILD_PROTOBUF)
    add_definitions(-DcPROTO)

    # Link library
    if(Protobuf_ROOT)
        # Find libraries (need absolute paths)
        find_library(Protobuf_LIBRARY NAMES protobuf libprotobuf HINTS ${Protobuf_ROOT} PATHS ${Protobuf_ROOT} PATH_SUFFIXES "lib" "lib64")
        find_library(Protobuf_LIBRARY_DEBUG NAMES protobufd libprotobufd HINTS ${Protobuf_ROOT} PATHS ${Protobuf_ROOT} PATH_SUFFIXES "lib" "lib64")
        find_library(Protobuf_LIBRARY_RELEASE NAMES protobuf libprotobuf HINTS ${Protobuf_ROOT} PATHS ${Protobuf_ROOT} PATH_SUFFIXES "lib" "lib64")
    else()
        find_package(Protobuf)  # Problems with: Protobuf_ROOT

        # Check if Protobuf was really found
        if(NOT Protobuf_FOUND)
            message(FATAL_ERROR "Protobuf was found by CMake.
                    Use '-D BUILD_SUPERBUILD=ON', or try with a different Protobuf installation to fix this problem.
                    Alternatively, you can disable it with '-D BUILD_PROTOBUF=OFF")
        endif()
    endif()

    # Add extra files
    target_sources(${PROJECT_NAME} PRIVATE ${ONNX_FILES})

    # Add includes
    target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${Protobuf_INCLUDE_DIRS}>)

    # Add libraries
    if(MSVC)
        target_link_libraries(${PROJECT_NAME} PUBLIC optimized ${Protobuf_LIBRARY} debug ${Protobuf_LIBRARY_DEBUG})
    else()
        target_link_libraries(${PROJECT_NAME} PUBLIC ${Protobuf_LIBRARY})
    endif()

    # Create "onnx.pb.cc" and "onnx.pb.h" files (from serialization/onnx)
    # Equivalent to: /usr/local/bin/protoc --cpp_out . onnx.proto
    #if(OVERWRITE_PROTO_FILES)
    #    EXEC_PROGRAM(${Protobuf_PROTOC_EXECUTABLE} ARGS --cpp_out ${PROJECT_SOURCE_DIR}/src/serialization/onnx/ --proto_path ${PROJECT_SOURCE_DIR}/src/serialization/onnx/ onnx.proto )
    #endif()
endif()


# OpenMP
if(BUILD_OPENMP)
    # find_package does not support Clang on Windows
    if (WIN32 AND CMAKE_RC_COMPILER STREQUAL "llvm-rc")
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp")
        set(OpenMP_C_FLAGS "-Xclang -fopenmp")

        # Tell CMake where to find the OpenMP libraries.
        get_filename_component(OpenMP_BIN_DIR ${CMAKE_CXX_COMPILER_AR} DIRECTORY)
        get_filename_component(OpenMP_DIR ${OpenMP_BIN_DIR} DIRECTORY)
        target_link_directories(${PROJECT_NAME} PUBLIC "${OpenMP_DIR}/lib")  # "The command will apply only to targets created after it is called."
        target_link_libraries(${PROJECT_NAME} PUBLIC libomp libiomp5md)

#        find_library(OMP_LIBRARY libomp)
#        find_library(OMP5MD_LIBRARY libiomp5md)
#        target_link_libraries(${PROJECT_NAME} PUBLIC ${OMP_LIBRARY} ${OMP5MD_LIBRARY})

    else()
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            # OpenMP added through a flag (OpenMP_CXX_FLAGS=="-fopenmp")
            SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

            if(${OpenMP_CXX_VERSION_MAJOR})
                set(OpenMP_VERSION_MAJOR ${OpenMP_CXX_VERSION_MAJOR} CACHE INTERNAL "" FORCE)
            endif()
            target_compile_definitions(${PROJECT_NAME} PUBLIC OpenMP_VERSION_MAJOR=${OpenMP_VERSION_MAJOR})

            set(OMP_CANCELLATION true)  # https://www.openmp.org/spec-html/5.0/openmpse59.html
            SET(USE_OPENMP ON) # Local's scope
        else()
            message(WARNING "OpenMP not found but requested during compilation. (Falling back to: '-DBUILD_OPENMP=OFF')\n\
            Hint: Use a compiler with OpenMP support using: '-D CMAKE_CXX_COMPILER=path'")
        endif()
    endif()
endif()
SET(USE_OPENMP ${USE_OPENMP} PARENT_SCOPE)  # Parent's scope


# CUDA
if(USE_CUDA)
    cmake_minimum_required(VERSION 3.17.2)  # Due to CUDAToolkit

    # Check if cuda is available
    include(CheckLanguage)
    check_language(CUDA)

    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)

        # NVCC needs GCC versions to be less or equal than 8  (GCC < 9.0;)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
                message(FATAL_ERROR "[WARNING] The nvcc compiler in CUDA 10 (or later) does not support gcc versions later than 8 (DETECTED: ${CMAKE_CXX_COMPILER_VERSION}).
        Hint: Use g++-8 (or older); Set other compiler version by using '-D CMAKE_CXX_COMPILER='$(which g++-8)' or creating a symbolic link.")
            endif()
        endif()

        # Set standard CUDA variables
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 11)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif()

        # Target properties
        set_target_properties(${PROJECT_NAME} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
                )

        # Add source files
        target_sources(${PROJECT_NAME} PRIVATE ${CUDA_HEADERS} ${CUDA_SOURCES})

        # Add includes
        target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

        # Add libraries
        target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cublas CUDA::cudart CUDA::curand)
        if(USE_CUDNN)
            if(CUDNN_ROOT_DIR)
                SET(CUDNN_INCLUDE_DIRS ${CUDNN_ROOT_DIR}/include)
                find_library(CUDNN_LIBRARY cudnn HINTS ${CUDNN_ROOT_DIR} PATHS ${CUDNN_ROOT_DIR} PATH_SUFFIXES "lib" "lib64")
            else()
                SET(CUDNN_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
                find_library(CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR} PATHS ${CUDAToolkit_LIBRARY_DIR})
            endif()

            # Check if the library has been found
            if(CUDNN_LIBRARY)
                target_include_directories(${PROJECT_NAME} PRIVATE ${CUDNN_INCLUDE_DIRS})
                target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDNN_LIBRARY})
            else()
                message(WARNING "[WARNING] CUDNN was not found but requested during compilation. (Falling back to: '-D BUILD_TARGET=GPU')
                Hint: Install CUDNN in the same path as the CUDA Toolkit, or specify the CUDNN path using this flag '-D CUDNN_ROOT_DIR=path'")
                SET(BUILD_TARGET "GPU") # Local's scope
                SET(USE_CUDNN OFF) # Local's scope (disable)
            endif()
        endif()

        if(APPLE)
            # We need to add the path to the driver (libcuda.dylib) as an rpath,
            # so that the static cuda runtime can find it at runtime.
            set_property(TARGET ${PROJECT_NAME} PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        endif()

        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            # Make EDDL works for cuda 7.5
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__")
        endif()
    else()
        message(WARNING "[WARNING] CUDA compiler not found but requested during compilation. (Falling back to: '-D USE_CUDA=OFF')
        Hint: Set the NVCC path using '-D CMAKE_CUDA_COMPILER=path' or creating a symbolic link.")
        SET(BUILD_TARGET "CPU") # Local's scope
        SET(USE_CUDNN OFF) # Local's scope (disable)
        SET(USE_CUDA OFF) # Local's scope (disable)
    endif()
endif()
SET(USE_CUDA ${USE_CUDA} PARENT_SCOPE)  # Parent's scope
SET(USE_CUDNN ${USE_CUDNN} PARENT_SCOPE)  # Parent's scope



# FPGA
if(USE_FPGA)
    find_package(OpenCL)
    if (OpenCL_FOUND)

        target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${OpenCL_INCLUDE_DIR};$ENV{XILINX_VIVADO}/include}>")
        #target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:/opt/Xilinx/Vivado/2018.3/include)  # Is this cross-platform?

        target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCL_LIBRARY})
        if(APPLE)
            target_link_options(${PROJECT_NAME} PUBLIC -framework opencl)
        endif()

        # TODO: Why this? Flags should be specific
        #set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS}")  # DO NOT REPLACE "CMAKE_CXX_FLAGS
        set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS})

        # Add extra files
        target_sources(${PROJECT_NAME} PRIVATE ${FPGA_HEADERS} ${FPGA_SOURCES})

    else()
        message(WARNING "OpenCL not found but requested during compilation. (Falling back to: '-D BUILD_TARGET=CPU')
        Hint: Set the OpenCL path using '-D OPENCL_ROOT=path' or creating a symbolic link.")
        SET(BUILD_TARGET "CPU") # Local's scope
        SET(USE_FPGA OFF) # Local's scope (disable)
    endif()
endif()
SET(USE_FPGA ${USE_FPGA} PARENT_SCOPE)  # Parent's scope

if(BUILD_DIST)
    # ZLIB
    if(DEFINED ZLIB_ROOT AND DEFINED ZLIB_INCLUDE_DIRS)
        find_library(ZLIB_LIBRARIES z HINTS ${ZLIB_ROOT} PATHS ${ZLIB_ROOT} PATH_SUFFIXES "lib" "lib64")
    else()
        find_package(ZLIB)

        # Check if ZLIB was really found
        if(NOT ZLIB_FOUND)
            message(FATAL_ERROR "ZLIB was not found by CMake.
                Use '-D BUILD_SUPERBUILD=ON', or try with a different ZLIB installation to fix this problem.")
        endif()
    endif()
    target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${ZLIB_INCLUDE_DIRS}>)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${ZLIB_LIBRARIES})

    # OPENSSL
    if(DEFINED OPENSSL_ROOT_DIR AND DEFINED OPENSSL_INCLUDE_DIR)
        find_library(OPENSSL_SSL_LIBRARY ssl HINTS ${OPENSSL_ROOT_DIR} PATHS ${OPENSSL_ROOT_DIR} PATH_SUFFIXES "lib" "lib64")
        find_library(OPENSSL_CRYPTO_LIBRARY crypto HINTS ${OPENSSL_ROOT_DIR} PATHS ${OPENSSL_ROOT_DIR} PATH_SUFFIXES "lib" "lib64")
        SET(OPENSSL_LIBRARIES ${OPENSSL_SSL_LIBRARY} ${OPENSSL_CRYPTO_LIBRARY})
    else()
        find_package(OpenSSL)

        # Check if ZLIB was really found
        if(NOT OPENSSL_FOUND)
            message(FATAL_ERROR "OpenSSL was not found by CMake.
                Use '-D BUILD_SUPERBUILD=ON', or try with a different OpenSSL installation to fix this problem.")
        endif()
    endif()
    target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${OPENSSL_INCLUDE_DIR}>)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENSSL_LIBRARIES})
endif()

###########################################################################
################################## WINDOWS ################################
###########################################################################

# Microsoft Visual C++ - set static/dynamic runtime
if(MSVC)
    if(BUILD_SHARED_LIBS)
        message(STATUS "Setting dynamic runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
            set(${flag_var} "${${flag_var}}" PARENT_SCOPE)
        endforeach(flag_var)
        if (USE_CUDA AND CMAKE_CUDA_COMPILER)
            foreach(flag_var CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE CMAKE_CUDA_FLAGS_MINSIZEREL CMAKE_CUDA_FLAGS_RELWITHDEBINFO)
                string(REGEX REPLACE "-MT" "-MD" ${flag_var} "${${flag_var}}")
                set(${flag_var} "${${flag_var}}" PARENT_SCOPE)
            endforeach(flag_var)
        endif()
    else()
        message(STATUS "Setting static runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
            set(${flag_var} "${${flag_var}}" PARENT_SCOPE)
        endforeach(flag_var)
        if (USE_CUDA AND CMAKE_CUDA_COMPILER)
            foreach(flag_var CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE CMAKE_CUDA_FLAGS_MINSIZEREL CMAKE_CUDA_FLAGS_RELWITHDEBINFO)
                string(REGEX REPLACE "-MD" "-MT" ${flag_var} "${${flag_var}}")
                set(${flag_var} "${${flag_var}}" PARENT_SCOPE)
            endforeach(flag_var)
        endif()
    endif()
endif()

##########################################################################
############################# DEFINITIONS ################################
##########################################################################

if(USE_CUDA)
    add_definitions(-DcGPU)
endif()
if(USE_CUDNN)
    add_definitions(-DcCUDNN)
endif()
if(USE_FPGA)
    add_definitions(-DcFPGA)
endif()

##########################################################################
############################ INSTALLATION ################################
##########################################################################

add_library(EDDL::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# Install
install(TARGETS ${PROJECT_NAME}
        EXPORT EDDLTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )

# Add EDDL includes
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT devel)

# SHOW OPTIONS
message(STATUS "===========================================" )
message(STATUS "============== CMAKE SUMMARY ==============" )
message(STATUS "===========================================" )
message(STATUS "Project name: " ${CMAKE_PROJECT_NAME} )
message(STATUS "Project version: " ${CMAKE_PROJECT_VERSION} )
message(STATUS "Prefix path: " ${CMAKE_PREFIX_PATH} )
message(STATUS "Install prefix: " ${CMAKE_INSTALL_PREFIX} )
message(STATUS "-------------------------------------------" )
message(STATUS "Build target: " ${BUILD_TARGET} )
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE} )
message(STATUS "Build shared libs: " ${BUILD_SHARED_LIBS} )
message(STATUS "Build coverage: " ${BUILD_COVERAGE} )
message(STATUS "Build sanitizers: " ${BUILD_SANITIZERS} )
message(STATUS "Build HPC: " ${BUILD_HPC} )
message(STATUS "Build distributed: " ${BUILD_DIST} )
message(STATUS "-------------------------------------------" )
message(STATUS "Find library suffixes: " ${CMAKE_FIND_LIBRARY_SUFFIXES} )
message(STATUS "-------------------------------------------" )
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER}) | Version: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "C++ flags (release): " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "C++ flags (debug): " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "-------------------------------------------" )
message(STATUS "OpenMP enabled: " ${USE_OPENMP} )
if(USE_OPENMP)
    message(STATUS "OpenMP gomp library: " ${OpenMP_gomp_LIBRARY} )
    message(STATUS "OpenMP pthread library: " ${OpenMP_pthread_LIBRARY} )
    message(STATUS "OpenMP flags: " ${OpenMP_CXX_FLAGS} )
endif()
message(STATUS "-------------------------------------------" )
message(STATUS "CUDA enabled: " ${USE_CUDA} )
if(USE_CUDA)
    message(STATUS "CUDA toolkit root dir: ${CUDAToolkit_LIBRARY_ROOT}")
    message(STATUS "CUDA toolkit include dir: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA toolkit include dir (targets): ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    message(STATUS "CUDA toolkit library dir: ${CUDAToolkit_LIBRARY_DIR}")
    message(STATUS "CUDA NVCC executable: ${CUDAToolkit_NVCC_EXECUTABLE}")
    message(STATUS "CUDA toolkit implicit link directories: " ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER} | Version: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CUDA host compiler: " ${CMAKE_CUDA_HOST_COMPILER})
endif()
message(STATUS "-------------------------------------------" )
message(STATUS "CUDNN enabled: " ${USE_CUDNN} )
if(USE_CUDNN)
    message(STATUS "CuDNN root dir: ${CUDNN_ROOT_DIR}")
    message(STATUS "CuDNN include dir: ${CUDNN_INCLUDE_DIRS}")
    message(STATUS "CuDNN libraries: ${CUDNN_LIBRARY}")
endif()
message(STATUS "-------------------------------------------" )
message(STATUS "FPGA enabled: " ${USE_FPGA} )
if(USE_FPGA)
    message(STATUS "OpenCL version: " ${OpenCL_VERSION_MAJOR} )
    message(STATUS "OpenCL include: " ${OpenCL_INCLUDE_DIRS} )
    message(STATUS "OpenCL libraries: " ${OpenCL_LIBRARIES} )
endif()
message(STATUS "-------------------------------------------" )
message(STATUS "Eigen3 root dir: " ${Eigen3_DIR} )
message(STATUS "Eigen3 include: " ${EIGEN3_INCLUDE_DIR} )
message(STATUS "-------------------------------------------" )
message(STATUS "Use Protobuf: " ${BUILD_PROTOBUF} )
if(BUILD_PROTOBUF)
    message(STATUS "Protobuf root dir: " ${Protobuf_ROOT} )
    message(STATUS "Protobuf include dirs: " ${Protobuf_INCLUDE_DIRS} )
    message(STATUS "Protobuf libraries: " ${Protobuf_LIBRARY} )
#    message(STATUS "Protobuf libraries (debug): " ${Protobuf_LIBRARY_DEBUG} )
#    message(STATUS "Protobuf libraries (release): " ${Protobuf_LIBRARY_RELEASE} )
    message(STATUS "Protobuf compiler: " ${Protobuf_PROTOC_EXECUTABLE} )
endif()
if(BUILD_DIST)
    message(STATUS "-------------------------------------------" )
    message(STATUS "ZLIB root: " ${ZLIB_ROOT} )
    message(STATUS "ZLIB include: " ${ZLIB_INCLUDE_DIRS} )
    message(STATUS "ZLIB libraries: " ${ZLIB_LIBRARIES} )
    message(STATUS "-------------------------------------------" )
    message(STATUS "OpenSSL root: " ${OPENSSL_ROOT_DIR} )
    message(STATUS "OpenSSL include: " ${OPENSSL_INCLUDE_DIR} )
    message(STATUS "OpenSSL SSL library: " ${OPENSSL_SSL_LIBRARY} )
    message(STATUS "OpenSSL crypto library: " ${OPENSSL_CRYPTO_LIBRARY} )
    message(STATUS "OpenSSL version: " ${OPENSSL_VERSION} )
endif()
message(STATUS "-------------------------------------------" )
message(STATUS "===========================================" )
message(STATUS "===========================================" )
