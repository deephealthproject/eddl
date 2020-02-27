cmake_minimum_required(VERSION 3.9.2)
project(demo)

set(CMAKE_CXX_STANDARD 11)

find_package(eddl)

add_executable(main main.cpp)
target_link_libraries(main eddl)

