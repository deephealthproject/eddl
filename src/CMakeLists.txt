target_sources(EDDLL
        PRIVATE
        net.cpp
        net.h
        utils.cpp
        utils.h
        compserv.cpp
        compserv.h
        )

add_subdirectory(layers)
add_subdirectory(cpu)

target_include_directories(EDDLL PUBLIC layers)
target_include_directories(EDDLL PUBLIC cpu)