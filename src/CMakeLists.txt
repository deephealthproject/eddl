###########################################################################
################################ OPTIONS ##################################
###########################################################################

# USER OPTIONS *******************************
set(BUILD_TARGET "GPU" CACHE STRING "Compile library for {CPU, GPU, FPGA}")
option(OVERWRITE_PROTO_FILES "Overwrite Protobuf files (requires a compatible Protobuf compiler)" ON)

# Double checks (restricted args)
set_property(CACHE BUILD_TARGET PROPERTY STRINGS CPU GPU FPGA)

# Initializations (Local's scope)
SET(USE_OPENMP OFF)
SET(USE_CUDA OFF)

# Device specific setup
string(TOUPPER ${BUILD_TARGET} BUILD_TARGET)  # Detect cpu, Cpu, CPU,...
if(${BUILD_TARGET} STREQUAL "CPU")
elseif(${BUILD_TARGET} STREQUAL "GPU")
    SET(USE_CUDA ON) # Local's scope
else()
    message(FATAL_ERROR "Build target '${BUILD_TARGET}' not implemented, CMake will exit." )
endif()

# Force default build type to Release
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")


###########################################################################
############################# SET LIBRARY #################################
###########################################################################

# C++ files
file(GLOB_RECURSE PUBLIC_HEADERS "${PROJECT_SOURCE_DIR}/include/*" *.{h, hpp})
file(GLOB_RECURSE CPP_HEADERS "${PROJECT_SOURCE_DIR}/src/*" *.{h, hpp})
file(GLOB_RECURSE CPP_SOURCES "${PROJECT_SOURCE_DIR}/src/*" *.{cc, cpp})
file(GLOB_RECURSE CUDA_SOURCES "${PROJECT_SOURCE_DIR}/src/*" *.{cu})

# Make an automatic library - will be static or dynamic based on user setting
add_library(${PROJECT_NAME} ${PUBLIC_HEADERS} ${CPP_HEADERS} ${CPP_SOURCES})

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

# Threads
if(UNIX) # Add setup for windows in the windows's section
    SET(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    SET(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads) 
endif()

## Eigen (I have temporally drop this header-only library to "includes/eddl/")
if(DEFINED Eigen3_DIR)
    message(STATUS "Using specific Eigen")
else()
    message(STATUS "Searching for Eigen in the system...")
endif()
find_package(Eigen3 3.3 REQUIRED NO_MODULE)  # EIGEN_DIR => ok
message(STATUS "Eigen found!")
target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)  # Header only library

# ZLIB
if(BUILD_SUPERBUILD AND NOT USE_SYSTEM_ZLIB)
    message(STATUS "Using specific ZLIB")
    SET(CMAKE_PREFIX_PATH "${ZLIB_BIN};${CMAKE_PREFIX_PATH}")
else()
    message(STATUS "Searching for ZLIB in the system...")
endif()

find_package(ZLIB REQUIRED)  # Problems with: ZLIB_ROOT
# Check if Protobuf was really found
if(NOT ZLIB_FOUND)
    message(FATAL_ERROR "ZLIB was found by CMake but its libraries or includes are missing.
            Use '-D BUILD_SUPERBUILD=ON', or try with a different ZLIB installation to fix this problem.")
endif()
message(STATUS "ZLIB found!")
target_link_libraries(${PROJECT_NAME} PUBLIC ZLIB::ZLIB)
# target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${ZLIB_INCLUDE_DIRS}>)
# target_link_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${ZLIB_LIBRARIES_DIRS}>)
# target_link_libraries(${PROJECT_NAME} PRIVATE ${ZLIB_LIBRARIES})

# ONNX files
if(BUILD_PROTOBUF)
    add_definitions(-DcPROTO)

    # Link library
    if(BUILD_SUPERBUILD AND NOT USE_SYSTEM_PROTOBUF)
        message(STATUS "Using specific Protobuf")
        SET(CMAKE_PREFIX_PATH "${Protobuf_ROOT}/bin;${CMAKE_PREFIX_PATH}")
        find_package(Protobuf CONFIG REQUIRED HINTS ${Protobuf_DIR})  # Problems with: Protobuf_DIR
        target_link_libraries(${PROJECT_NAME} PRIVATE protobuf::libprotobuf protobuf::libprotoc)
    else()
        message(STATUS "Searching for Protobuf in the system...")
        find_package(Protobuf REQUIRED)  # Problems with: Protobuf_DIR
        message(STATUS "Protobuf found! (system)")

        # Check if Protobuf was really found
        if(NOT Protobuf_FOUND)
            message(FATAL_ERROR "Protobuf was found by CMake but its libraries or includes are missing.
                    Use '-D BUILD_SUPERBUILD=ON', or try with a different Protobuf installation to fix this problem.
                    Alternatively, you can disable it with '-D BUILD_PROTOBUF=OFF")
        endif()
        target_include_directories(${PROJECT_NAME} PUBLIC ${Protobuf_INCLUDE_DIRS})
        target_link_directories(${PROJECT_NAME} PUBLIC ${Protobuf_LIBRARIES_DIRS})
        target_link_libraries(${PROJECT_NAME} PRIVATE ${Protobuf_LIBRARIES})
    endif()

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
        set(OMP_CANCELLATION true)  # https://www.openmp.org/spec-html/5.0/openmpse59.html
        SET(USE_OPENMP ON) # Local's scope
    else()
        message(WARNING "OpenMP not found but requested during compilation. (Falling back to: '-D BUILD_OPENMP=OFF')
        Hint: Use a compiler with OpenMP support using: '-D CMAKE_CXX_COMPILER=path'")
    endif()
endif()
SET(USE_OPENMP ${USE_OPENMP} PARENT_SCOPE)  # Parent's scope


# CUDA
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
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

        target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
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
        message(WARNING "[WARNING] CUDA compiler not found but requested during compilation. (Falling back to: '-D USE_CUDA=OFF')
        Hint: Set the NVCC path using '-D CMAKE_CUDA_COMPILER=path' or creating a symbolic link.")
        SET(BUILD_TARGET "CPU") # Local's scope
        SET(USE_CUDA OFF) # Local's scope (disable)
    endif()
endif()
SET(USE_CUDA ${USE_CUDA} PARENT_SCOPE)  # Parent's scope

###########################################################################
################################## WINDOWS ################################
###########################################################################

if(WIN32)
   # Threads *************************************************************
   if(PTHREADS_INSTALL_PATH STREQUAL "")
       message(ERROR ": Pthreads install path must be specified under Windows")
   endif()
   target_link_libraries(${PROJECT_NAME} PRIVATE debug ${PTHREADS_INSTALL_PATH}/lib/libpthreadVC3d.lib
           PRIVATE debug ${PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3d.lib
           PRIVATE debug ${PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3d.lib)
   target_link_libraries(${PROJECT_NAME} PRIVATE optimized ${PTHREADS_INSTALL_PATH}/lib/libpthreadVC3.lib
           PRIVATE optimized ${PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3.lib
           PRIVATE optimized ${PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3.lib)
   target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${PTHREADS_INSTALL_PATH}/include>)
endif()

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

###########################################################################
############################# INSTALLATION ################################
###########################################################################

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
message(STATUS "Superbuild: " ${BUILD_SUPERBUILD} )
message(STATUS "-------------------------------------------" )
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "C++ flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "C++ flags (release): " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "C++ flags (debug): " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "-------------------------------------------" )
message(STATUS "CUDA enabled: " ${USE_CUDA} )
if(USE_CUDA)
message(STATUS "CUDA compiler: " ${CMAKE_CUDA_COMPILER})
message(STATUS "CUDA flags: " ${CMAKE_CUDA_FLAGS} )
message(STATUS "CUDA toolkit: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )
endif()
message(STATUS "-------------------------------------------" )
message(STATUS "OpenMP enabled: " ${USE_OPENMP} )
if(BUILD_OPENMP)
message(STATUS "OpenMP version: " ${OpenMP_VERSION_MAJOR} )
message(STATUS "OpenMP gomp library: " ${OpenMP_gomp_LIBRARY} )
message(STATUS "OpenMP pthread library: " ${OpenMP_pthread_LIBRARY} )
endif()
message(STATUS "-------------------------------------------" )
message(STATUS "Eigen3 dir: " ${Eigen3_DIR} )
message(STATUS "Eigen3 include: " ${EIGEN3_INCLUDE_DIR} )
message(STATUS "-------------------------------------------" )
message(STATUS "ZLIB root: " ${ZLIB_ROOT} )
message(STATUS "ZLIB include: " ${ZLIB_INCLUDE_DIRS} )
message(STATUS "ZLIB libraries: " ${ZLIB_LIBRARIES} )
message(STATUS "-------------------------------------------" )
message(STATUS "Use Protobuf: " ${BUILD_PROTOBUF} )
if(BUILD_PROTOBUF)
message(STATUS "Protobuf dir: " ${Protobuf_DIR} )
message(STATUS "Protobuf include: " ${Protobuf_INCLUDE_DIRS} )
message(STATUS "Protobuf libraries: " ${Protobuf_LIBRARIES} )
message(STATUS "Protobuf compiler: " ${Protobuf_PROTOC_EXECUTABLE} )
endif()
if(WIN32)
message(STATUS "-------------------------------------------" )
message(STATUS "Pthreads dir: " ${PTHREADS_INSTALL_PATH} )
message(STATUS "Pthreads include: " ${PTHREADS_INSTALL_PATH}/include )
message(STATUS "Pthreads libraries: " ${PTHREADS_INSTALL_PATH}/lib )
endif()
message(STATUS "===========================================" )
message(STATUS "===========================================" )
