cmake_minimum_required(VERSION 3.9.2)

project(eddl-tests)

# Find tests (recursively)
file(GLOB_RECURSE CPP_TESTS_FILES LIST_DIRECTORIES true *.h *.cpp)

# GTest
enable_testing()
find_package(GTest REQUIRED)  # GTEST_ROOT

# Build test and target libraries
add_executable(unit_tests ${CPP_TESTS_FILES})
target_include_directories(unit_tests PUBLIC eddl ${GTEST_INCLUDE_DIRS})
target_link_libraries(unit_tests PUBLIC eddl ${GTEST_BOTH_LIBRARIES})

# Add test
add_test(NAME unit_tests COMMAND unit_tests)