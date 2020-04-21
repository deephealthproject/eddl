cmake_minimum_required(VERSION 3.9.2)

project(eddl-tests)

# GTest (use local)
enable_testing()
if(DEFINED GTEST_INCLUDE_DIRS AND DEFINED GTEST_LIBRARIES_DIRS AND DEFINED GTEST_BOTH_LIBRARIES)
    message(STATUS "Using specific GTest")
else()
    message(STATUS "Searching for GTest in the system...")
    find_package(GTest REQUIRED)
    message(STATUS "GTest found!")
endif()

# Find tests (recursively, from here)
file(GLOB_RECURSE CPP_TESTS_FILES "${PROJECT_SOURCE_DIR}/*" *.{h, cpp})

# Build test and target libraries
add_executable(unit_tests ${CPP_TESTS_FILES})
target_link_directories(unit_tests PUBLIC ${GTEST_LIBRARIES_DIRS})
target_link_libraries(unit_tests PUBLIC eddl ${GTEST_BOTH_LIBRARIES})

# What we should do:
target_include_directories(unit_tests PUBLIC ${GTEST_INCLUDE_DIRS})
## Really dirty trick!!! I do this to access private methods
#target_include_directories(unit_tests
#        PUBLIC
#            ${GTEST_INCLUDE_DIRS}
#        PRIVATE
#            ${PROJECT_SOURCE_DIR}/../include
#            ${PROJECT_SOURCE_DIR}/../src
#        )

# Add test
add_test(NAME unit_tests COMMAND unit_tests)


# SHOW OPTIONS
message(STATUS "===========================================" )
message(STATUS "===========================================" )
message(STATUS "GTest root: " ${GTEST_ROOT} )
message(STATUS "GTest include: " ${GTEST_INCLUDE_DIRS} )
message(STATUS "GTest libraries: " ${GTEST_BOTH_LIBRARIES} )
message(STATUS "===========================================" )
message(STATUS "===========================================" )
