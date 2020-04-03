include(ExternalProject)
ExternalProject_Add(protobuf
        PREFIX protobuf
        GIT_REPOSITORY https://github.com/google/protobuf
        GIT_TAG "v3.11.4"
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf-src"
        SOURCE_SUBDIR cmake
        CMAKE_CACHE_ARGS
            -Dprotobuf_BUILD_TESTS:BOOL=OFF
            -Dprotobuf_WITH_ZLIB:BOOL=OFF
            -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF
            -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/protobuf
        )

set(Protobuf_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf/lib/cmake/protobuf")
set(Protobuf_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/protobuf/include")
set(Protobuf_LIBRARIES_DIRS "${CMAKE_CURRENT_BINARY_DIR}/protobuf/lib")
set(Protobuf_LIBRARIES "protobuf")
set(Protobuf_PROTOC_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/protobuf/bin/protoc")