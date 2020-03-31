cmake_minimum_required(VERSION 3.9.2)

PROJECT(eddllib LANGUAGES CXX)


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

    # Find protobuf
    if(TARGET protobuf::libprotobuf)
        include(FindProtobuf)
        find_package(Protobuf REQUIRED)
        add_definitions(-DcPROTO)
        target_include_directories(eddl PUBLIC ${PROTOBUF_INCLUDE_DIR})
        target_link_libraries(eddl PRIVATE ${PROTOBUF_LIBRARIES})
        message("Protobuf path: ${PROTOBUF_LIBRARIES} ${PROTOBUF_INCLUDE_DIR}")
    else()
        add_definitions(-DcPROTO)
        target_include_directories(${PROJECT_NAME} PUBLIC ${PROTOBUF_INCLUDE_DIR})
        target_link_libraries(${PROJECT_NAME} PRIVATE ${PROTOBUF_LIB_DIR})
    endif()

    # Set files
    file(GLOB_RECURSE ONNX_HEADER_FILES
            ${PROJECT_SOURCE_DIR}/serialization/onnx/*.h
            )
    file(GLOB_RECURSE ONNX_SOURCES_FILES
            ${PROJECT_SOURCE_DIR}/serialization/onnx/*.cpp
            ${PROJECT_SOURCE_DIR}/serialization/onnx/*.cc
            )
    SET(ONNX_SOURCES ${ONNX_HEADER_FILES} ${ONNX_SOURCES_FILES})
endif()



# C++ files
file(GLOB CPP_HEADER_FILES_TOP
        ${PROJECT_SOURCE_DIR}/*.h
        )
file(GLOB CPP_SOURCE_FILES_TOP
        ${PROJECT_SOURCE_DIR}/*.cpp
        )
file(GLOB_RECURSE CPP_HEADER_FILES_NESTED
        ${PROJECT_SOURCE_DIR}/apis/*.h
        ${PROJECT_SOURCE_DIR}/descriptors/*.h
        ${PROJECT_SOURCE_DIR}/hardware/cpu/*.h
        ${PROJECT_SOURCE_DIR}/initializers/*.h
        ${PROJECT_SOURCE_DIR}/layers/*.h
        ${PROJECT_SOURCE_DIR}/losses/*.h
        ${PROJECT_SOURCE_DIR}/metrics/*.h
        ${PROJECT_SOURCE_DIR}/net/*.h
        ${PROJECT_SOURCE_DIR}/optimizers/*.h
        ${PROJECT_SOURCE_DIR}/regularizers/*.h
        ${PROJECT_SOURCE_DIR}/tensor/*.h
        )

file(GLOB_RECURSE CPP_SOURCE_FILES_NESTED
        ${PROJECT_SOURCE_DIR}/apis/*.cpp
        ${PROJECT_SOURCE_DIR}/descriptors/*.cpp
        ${PROJECT_SOURCE_DIR}/hardware/cpu/*.cpp
        ${PROJECT_SOURCE_DIR}/initializers/*.cpp
        ${PROJECT_SOURCE_DIR}/layers/*.cpp
        ${PROJECT_SOURCE_DIR}/losses/*.cpp
        ${PROJECT_SOURCE_DIR}/metrics/*.cpp
        ${PROJECT_SOURCE_DIR}/net/*.cpp
        ${PROJECT_SOURCE_DIR}/optimizers/*.cpp
        ${PROJECT_SOURCE_DIR}/regularizers/*.cpp
        ${PROJECT_SOURCE_DIR}/tensor/*.cpp
        )
SET(CPP_HEADER_FILES ${CPP_HEADER_FILES_TOP} ${CPP_HEADER_FILES_NESTED})
SET(CPP_SOURCE_FILES ${CPP_SOURCE_FILES_TOP} ${CPP_SOURCE_FILES_NESTED})
SET(CPP_SOURCES ${CPP_HEADER_FILES} ${CPP_SOURCE_FILES})


# CUDA files
file(GLOB_RECURSE CUDA_HEADER_FILES
        ${PROJECT_SOURCE_DIR}/hardware/gpu/*.h
        )
file(GLOB_RECURSE CUDA_CU_FILES
        ${PROJECT_SOURCE_DIR}/hardware/gpu/*.cu
        )
SET(CUDA_SOURCES ${CUDA_HEADER_FILES} ${CUDA_CU_FILES})


# ONNX files
SET(ONNX_SOURCES "")
if(BUILD_PROTOBUF)
    file(GLOB_RECURSE ONNX_HEADER_FILES
            ${PROJECT_SOURCE_DIR}/serialization/onnx/*.h
            )
    file(GLOB_RECURSE ONNX_SOURCES_FILES
            ${PROJECT_SOURCE_DIR}/serialization/onnx/*.cpp
            ${PROJECT_SOURCE_DIR}/serialization/onnx/*.cc
            )
    SET(ONNX_SOURCES ${ONNX_HEADER_FILES} ${ONNX_SOURCES_FILES})
endif()


# [LIBRARY] EDDL LIBRARY *********************
if(BUILD_SHARED_LIB)
    add_library(${PROJECT_NAME} SHARED "")
else(BUILD_SHARED_LIB)
    add_library(${PROJECT_NAME} STATIC "")
endif(BUILD_SHARED_LIB)

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
if(TARGET Eigen3::Eigen)
    find_package(Eigen3 3.3)
    target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
else()
    target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()


# ZLIB
if(TARGET ZLIB::ZLIB)
    find_package(ZLIB)
    target_include_directories(${PROJECT_NAME} PUBLIC ${ZLIB_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ZLIB_LIBRARIES})
else()
    target_include_directories(${PROJECT_NAME} PUBLIC ${ZLIB_INCLUDE_DIR})
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

# Install
install(TARGETS ${PROJECT_NAME}
        EXPORT CMakeExternalTargets
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        #RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )