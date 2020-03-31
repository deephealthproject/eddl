cmake_minimum_required(VERSION 3.9.2)

enable_testing()
find_package(GTest REQUIRED)

# Find tests (recursively)
file(GLOB_RECURSE CPP_TESTS_FILES LIST_DIRECTORIES true *.h *.cpp)

# Build test and target libraries
add_executable(unit_tests ${CPP_TESTS_FILES})
target_link_libraries(unit_tests PUBLIC eddl GTest::GTest GTest::Main)

# Add test
add_test(NAME unit_tests COMMAND unit_tests)