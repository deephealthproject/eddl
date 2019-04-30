target_sources(EDDLL
	PRIVATE
		eddl.cpp
		eddl.h
		loss.cpp
		loss.h
		metric.cpp
		metric.h
		net.cpp
		net.h
		optim.cpp
		optim.h
		tensor.cpp
		tensor.h
		tensor_op.cpp
		utils.cpp
		utils.h
        compserv.cpp
        compserv.h
)

add_subdirectory(layers)
add_subdirectory(cpu)

target_include_directories(EDDLL PUBLIC layers)
target_include_directories(EDDLL PUBLIC cpu)