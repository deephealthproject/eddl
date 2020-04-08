include(ExternalProject)
ExternalProject_Add(GTest
        PREFIX googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG "release-1.10.0"
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/googletest-src"
        CMAKE_CACHE_ARGS
            -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/googletest
        )

# Set variables
SET(GTEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}/googletest")
