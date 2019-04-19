# This file is part of ECVL an European Computer Vision Library.
# Developed within the DeepHealth project.
# Boosting AI in Europe.
#
# The MIT License (MIT)
#
# Copyright (c) 2019
#           Federico Bolelli, <federico.bolelli@unimore.it>
#           Costantino Grana, <costantino.grana@unimore.it>
#			Stefano Allegretti, <stefano.allegretti@unimore.it>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required (VERSION 3.13)

project (EDDLL)
add_library(EDDLL "")
add_subdirectory(src)

#--------------------------------------------------------------------------------------------------------
# BEGIN: static/dinamic runtime 												
#--------------------------------------------------------------------------------------------------------
set(EDDLL_STATIC ON CACHE BOOL "Whether to build static or dynamic libraries")

if(MSVC)
    if(EDDLL_STATIC)
        message(STATUS "Setting static runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO) 
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
    else()
	    message(STATUS "Setting dinamic runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO) 
            string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
	endif()
endif()
#--------------------------------------------------------------------------------------------------------
# END: static/dinamic runtime										
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# BEGIN: pthreads setup for windows os												
#--------------------------------------------------------------------------------------------------------

# The following commands should be removed when pthread linux 
# specific calls will be replaced by thread c++ standard calls 
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
	set(EDDLL_PTHREADS_INSTALL_PATH "" CACHE PATH "")
	
	if(EDDLL_PTHREADS_INSTALL_PATH STREQUAL "")
		message(ERROR " pthreds install path must be specified under Windows")
	endif()
	
	target_link_libraries(EDDLL debug "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVC3d.lib"
								debug "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3d.lib"
								debug "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3d.lib")
								 
	target_link_libraries(EDDLL optimized "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVC3.lib"
								optimized "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3d.lib"
								optimized "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3d.lib")
								 
	include_directories("${EDDLL_PTHREADS_INSTALL_PATH}/include")
	
	# The following command should be moved elsewhere
	if(MSVC)
		add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	endif()
endif()
#--------------------------------------------------------------------------------------------------------
# END: pthreads setup for windows os												
#--------------------------------------------------------------------------------------------------------

add_executable(MNIST-CONV "")
add_subdirectory(examples/MNIST/CONV)
target_link_libraries(MNIST-CONV EDDLL)

#--------------------------------------------------------------------------------------------------------
# BEGIN: install EDDLL (basic version)											
#--------------------------------------------------------------------------------------------------------
#set(${CMAKE_DEBUG_POSTFIX} "d")
install(TARGETS EDDLL
		CONFIGURATIONS Debug
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/Debug/bin
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/Debug/lib
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/Debug/lib/static
		)
install(TARGETS EDDLL
		CONFIGURATIONS Release
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/Release/bin
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/Release/lib
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/Release/lib/static
		)		
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/
		DESTINATION ${CMAKE_INSTALL_PREFIX}/include
        FILES_MATCHING PATTERN "*.h")

# TODO: use Eigen CMakeLists.txt instead of doing it by hand
set(EIGEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/Eigen")
set(ESCAPED_CMAKE_CURRENT_SOURCE_DIR "${EIGEN_SOURCE_DIR}")
file(GLOB Eigen_directory_files "${EIGEN_SOURCE_DIR}/*")		
#message(STATUS ${Eigen_directory_files})
foreach(f ${Eigen_directory_files})
  if(NOT f MATCHES "\\.txt" AND NOT f MATCHES "${ESCAPED_CMAKE_CURRENT_SOURCE_DIR}/[.].+" AND NOT f MATCHES "${ESCAPED_CMAKE_CURRENT_SOURCE_DIR}/src")
    list(APPEND Eigen_directory_files_to_install ${f})
  endif()
endforeach(f ${Eigen_directory_files})
#message(STATUS ${Eigen_directory_files_to_install})

install(FILES
  ${Eigen_directory_files_to_install}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/cpu/Eigen/
  )
		
#--------------------------------------------------------------------------------------------------------
# END: install EDDLL (basic version)												
#--------------------------------------------------------------------------------------------------------