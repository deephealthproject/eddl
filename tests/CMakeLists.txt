cmake_minimum_required(VERSION 3.9.2)

project(eddl-tests)

if(USE_LOCAL_GTEST AND NOT BUILD_SUPERBUILD)
    message(STATUS "Using local GTest...")
else()
    # GTest (use local)
    enable_testing()
    if(DEFINED GTEST_INCLUDE_DIRS AND DEFINED GTEST_LIBRARIES_DIRS AND DEFINED GTEST_BOTH_LIBRARIES)
        message(STATUS "Using specific GTest")
    else()
        message(STATUS "Searching for GTest in the system...")
        find_package(GTest REQUIRED)
        message(STATUS "GTest found!")
        SET(GTEST_BOTH_LIBRARIES "gtest;gtest_main")

        # Check if GTest was really found
        if(NOT GTEST_FOUND)
            message(FATAL_ERROR "GTest was found by CMake but its libraries or includes are missing.
                    Use '-D USE_LOCAL_GTEST=ON', -D BUILD_SUPERBUILD=ON, or try with a different GTest installation to
                    fix this problem. Alternatively, you can disable this tests with '-D BUILD_TESTS=OFF")
        endif()
    endif()
endif()

# Find tests (recursively, from here)
file(GLOB_RECURSE CPP_TESTS_FILES "${PROJECT_SOURCE_DIR}/*" *.{h, cpp})

# Build test and target libraries
add_executable(unit_tests ${CPP_TESTS_FILES})
target_include_directories(unit_tests PUBLIC $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>)  # TODO: Why build interface?
#target_link_directories(unit_tests PUBLIC ${GTEST_LIBRARIES_DIRS})
link_directories(${GTEST_LIBRARIES_DIRS})
if(MSVC)
    target_link_libraries(unit_tests PRIVATE eddl
        optimized ${GTEST_BOTH_LIBRARIES}
        debug ${GTEST_BOTH_LIBRARIES_D})
else()
    target_link_libraries(unit_tests PRIVATE eddl ${GTEST_BOTH_LIBRARIES})
endif()

# Add test
add_test(NAME unit_tests COMMAND unit_tests)


# SHOW OPTIONS
message(STATUS "===========================================" )
message(STATUS "===========================================" )
message(STATUS "Use local GTest: ${USE_LOCAL_GTEST} (different from superbuild)" )
message(STATUS "GTest root: " ${GTEST_ROOT} )
message(STATUS "GTest include: " ${GTEST_INCLUDE_DIRS} )
message(STATUS "GTest library dirs: " ${GTEST_LIBRARIES_DIRS} )
message(STATUS "GTest libraries: " ${GTEST_BOTH_LIBRARIES} )
message(STATUS "===========================================" )
message(STATUS "===========================================" )
