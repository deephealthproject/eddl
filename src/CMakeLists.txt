# [LIBRARY] EDDL LIBRARY *********************
if(BUILD_SHARED_LIB)
    add_library(${PROJECT_NAME} SHARED "")
else(BUILD_SHARED_LIB)
    add_library(${PROJECT_NAME} STATIC "")
endif(BUILD_SHARED_LIB)


# Add CUDA (if possible)
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__")
            set(CMAKE_CUDA_STANDARD 11)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif()
    else()
        set(USE_CUDA OFF)
        set(BUILD_TARGET "CPU")
    endif()
endif()


# ONNX files
SET(ONNX_SOURCES "")
if(BUILD_PROTOBUF)
    add_definitions(-DcPROTO)

    # Link library
#    target_include_directories(${PROJECT_NAME} PUBLIC ${PROTOBUF_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE libprotobuf libprotobuf-lite libprotoc)

#    if(Protobuf_FOUND)
#        message("Using system's protobuf")
#        #protobuf_generate_cpp(onnx.pb.cc onnx.pb.h onnx.proto) #Generate protobuf files #Doesn't work
#        # It seems that in some computers we the path appended to the file
#        EXEC_PROGRAM(${PROTOBUF_PROTOC_EXECUTABLE} ARGS --cpp_out ${PROJECT_SOURCE_DIR}/src/serialization/onnx/ --proto_path ${PROJECT_SOURCE_DIR}/src/serialization/onnx/ onnx.proto )
#
#        # Link library
#        target_include_directories(${PROJECT_NAME} PUBLIC ${PROTOBUF_INCLUDE_DIRS})
#        target_link_libraries(${PROJECT_NAME} PRIVATE ${PROTOBUF_LIBRARIES})
#    else()
#        message("Using project's protobuf")
#        SET(BUILD_PROTOBUF OFF)
#        message(WARNING "Protobuf not found, ignore if you are not using the ONNX module.")
#    endif()

    # Set files (Needs to be here since "onnx.pb.cc" and "onnx.pb.h" could have been generated above
    file(GLOB_RECURSE ONNX_HEADER_FILES
            serialization/onnx/*.h
            )
    file(GLOB_RECURSE ONNX_SOURCES_FILES
            serialization/onnx/*.cpp
            serialization/onnx/*.cc
            )
    SET(ONNX_SOURCES ${ONNX_HEADER_FILES} ${ONNX_SOURCES_FILES})
endif()



# C++ files
file(GLOB CPP_HEADER_FILES_TOP
        *.h
        )
file(GLOB CPP_SOURCE_FILES_TOP
        *.cpp
        )
file(GLOB_RECURSE CPP_HEADER_FILES_NESTED
        apis/*.h
        descriptors/*.h
        hardware/cpu/*.h
        initializers/*.h
        layers/*.h
        losses/*.h
        metrics/*.h
        net/*.h
        optimizers/*.h
        regularizers/*.h
        tensor/*.h
        )

file(GLOB_RECURSE CPP_SOURCE_FILES_NESTED
        apis/*.cpp
        descriptors/*.cpp
        hardware/cpu/*.cpp
        initializers/*.cpp
        layers/*.cpp
        losses/*.cpp
        metrics/*.cpp
        net/*.cpp
        optimizers/*.cpp
        regularizers/*.cpp
        tensor/*.cpp
        )
SET(CPP_HEADER_FILES ${CPP_HEADER_FILES_TOP} ${CPP_HEADER_FILES_NESTED})
SET(CPP_SOURCE_FILES ${CPP_SOURCE_FILES_TOP} ${CPP_SOURCE_FILES_NESTED})
SET(CPP_SOURCES ${CPP_HEADER_FILES} ${CPP_SOURCE_FILES})

# CUDA files
file(GLOB_RECURSE CUDA_HEADER_FILES
        hardware/gpu/*.h
        )
file(GLOB_RECURSE CUDA_CU_FILES
        hardware/gpu/*.cu
        )
SET(CUDA_SOURCES ${CUDA_HEADER_FILES} ${CUDA_CU_FILES})


target_sources(${PROJECT_NAME} PRIVATE ${CPP_SOURCES} ${ONNX_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        )

# pthreads setup for linux os
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    SET(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    SET(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
endif()

# Eigen
find_package (Eigen3 3.3 CONFIG REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

# ZLIB
find_package(ZLIB)
if(ZLIB_FOUND)
    message("Using system's ZLIB")
    target_include_directories(${PROJECT_NAME} PUBLIC ${ZLIB_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ZLIB_LIBRARIES})
else()
    message("Using project's ZLIB")
    target_include_directories(${PROJECT_NAME} PUBLIC ${ZLIB_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ZLIB_LIBRARY})
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
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    target_sources(${PROJECT_NAME} PRIVATE ${CUDA_SOURCES})
    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(${PROJECT_NAME}
            PRIVATE cublas
            PRIVATE cudart
            PRIVATE curand
            )
    add_definitions(-DcGPU)

    # Target properties
    set_target_properties(${PROJECT_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()


# TODO: REVIEW installtion paths (potential conflict with GNUInstallDirs)
# [INSTALL] Install the EDDL library
message(STATUS "Install path: ${CMAKE_INSTALL_PREFIX}")

if (WIN32)
    SET(EDDL_CONFIG_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
else()
    SET(EDDL_CONFIG_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/eddl)
endif()

install(TARGETS ${PROJECT_NAME}
        EXPORT eddlTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )
install(EXPORT eddlTargets
        DESTINATION ${EDDL_CONFIG_INSTALL_PATH}
        )
install(DIRECTORY .
        DESTINATION include/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.h")

# This allows to do: "find_package(eddl)"
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/eddlConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/../cmake/eddlConfig.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/../cmake/eddlConfig.cmake"
        DESTINATION ${EDDL_CONFIG_INSTALL_PATH}
        )

# SHOW OPTIONS
message(STATUS "===========================================" )
message(STATUS "===========================================" )
message(STATUS "Project name: " ${CMAKE_PROJECT_NAME} )
message(STATUS "Prefix path: " ${CMAKE_PREFIX_PATH} )
message(STATUS "Install prefix: " ${CMAKE_INSTALL_PREFIX} )
message(STATUS "-------------------------------------------" )
message(STATUS "Build target: " ${BUILD_TARGET} )
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE} )
message(STATUS "Build tests: " ${BUILD_TESTS} )
message(STATUS "Build examples: " ${BUILD_EXAMPLES} )
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
message(STATUS "Protobuf include: " ${PROTOBUF_INCLUDE_DIR} )
message(STATUS "Protobuf libraries: " ${PROTOBUF_LIBRARIES} )
message(STATUS "Protobuf compiler: " ${PROTOBUF_PROTOC_EXECUTABLE} )
message(STATUS "-------------------------------------------" )
message(STATUS "Eigen3 include: " ${EIGEN3_INCLUDE_DIR} )
message(STATUS "Eigen3 dir: " ${Eigen3_DIR} )
message(STATUS "-------------------------------------------" )
message(STATUS "ZLIB root: " ${ZLIB_ROOT} )
message(STATUS "ZLIB include: " ${ZLIB_INCLUDE_DIRS} )
message(STATUS "ZLIB libraries: " ${ZLIB_LIBRARIES} )
message(STATUS "===========================================" )
message(STATUS "===========================================" )
