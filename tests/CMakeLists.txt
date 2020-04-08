cmake_minimum_required(VERSION 3.9.2)

project(eddl-tests)

# GTest (use local)
enable_testing()
if(DEFINED GTEST_ROOT)
    message("Using specific GTest")
else()
    message("Searching for GTest in the system...")
endif()
find_package(GTest REQUIRED)
message("GTest found!")

# Find tests (recursively)
file(GLOB_RECURSE CPP_TESTS_FILES LIST_DIRECTORIES true *.h *.cpp)

# GTest
if(DEFINED GTEST_ROOT)
    message("Using specific GTest")
else()
    message("Searching for GTest in the system...")
endif()
find_package(GTest REQUIRED)  # Problems with: ZLIB_ROOT
message("GTest found!")

# Build test and target libraries
add_executable(unit_tests ${CPP_TESTS_FILES})
target_include_directories(unit_tests PUBLIC eddl ${GTEST_INCLUDE_DIRS})
target_link_libraries(unit_tests PUBLIC eddl ${GTEST_LIBRARIES})

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
