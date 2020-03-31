# Set name
set(TEST_NAME unit_tests)

# Find tests (recursively)
file(GLOB_RECURSE CPP_TESTS_FILES LIST_DIRECTORIES true *.h *.cpp)

message(STATUS "Test files: ${CPP_TESTS_FILES}")

# Build test and target libraries
add_executable(${TEST_NAME} ${CPP_TESTS_FILES})

# Add test
add_test(
        NAME
        ${TEST_NAME}
        COMMAND
        ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${TEST_NAME}
)

target_link_libraries(${TEST_NAME} PUBLIC ${PROJECT_NAME} ${GTEST_INCLUDE_DIRS})
