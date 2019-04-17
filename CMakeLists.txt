cmake_minimum_required (VERSION 3.13)

project (EDDLL)
add_library(EDDLL "")
add_subdirectory(src)

#--------------------------------------------------------------------------------------------------------
# BEGIN: static/dinamic runtime 												
#--------------------------------------------------------------------------------------------------------
set(EDDLL_STATIC ON CACHE BOOL "Whether to build static or dinamic libraries")

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


