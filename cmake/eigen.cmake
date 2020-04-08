include(ExternalProject)
ExternalProject_Add(Eigen3
		PREFIX eigen
		GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
		GIT_TAG "3.3.7"
		SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/eigen-src"
		CMAKE_CACHE_ARGS
			-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/eigen
		)

# Set variables
SET(Eigen3_DIR "${CMAKE_CURRENT_BINARY_DIR}/eigen/share/eigen3/cmake" )
SET(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/eigen/include/eigen3" )
