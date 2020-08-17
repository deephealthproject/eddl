cmake_minimum_required(VERSION 3.9.2)

project(eddl-examples)


# RUNTIME EXECUTABLES FOR AD HOC DISTRIBUTED VERSION: REQUIRES ONNX
if(BUILD_PROTOBUF)
    add_executable(master "distributed/master.cpp")
    target_link_libraries(master eddl)

    add_executable(worker "distributed/worker.cpp")
    target_link_libraries(worker eddl)

    add_executable(misc_info "distributed/misc_info.cpp")
    target_link_libraries(misc_info eddl)
endif()
