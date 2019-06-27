cmake_minimum_required(VERSION 3.9.2)
PROJECT(eddl)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")

# SET C++ COMPILER STANDARD
SET(CMAKE_CXX_STANDARD 11)

# SET COMPILER FLAGS
#set(CMAKE_CXX_FLAGS "-march=native -mavx -O3")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE} )
message(STATUS "C++ flags: " ${CMAKE_CXX_FLAGS} )


# SET STATIC/DYNAMIC RUNTIME
set(EDDLL_SHARED OFF CACHE BOOL "Whether to build a static or dynamic library")

if(MSVC)
    if(EDDLL_SHARED)
        message(STATUS "Setting dinamic runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
    else(EDDLL_SHARED)
        message(STATUS "Setting static runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
	endif(EDDLL_SHARED)

	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()


macro(eddll_update_third_party SUBMODULE)
    if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodule as needed
        message(STATUS "${SUBMODULE} update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive third_party/${SUBMODULE}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout ${SUBMODULE} submodule")
        endif()
endif()
endmacro()


# UPDATE THIRD_PARTY MANDATORY MODULES
find_package(Git QUIET)
eddll_update_third_party(eigen)
include_directories(third_party/eigen)

# USER OPTIONS *******************************
option(BUILD_PYTHON "Compile python binding" ON)
option(BUILD_TESTS "Compile tests" ON)
option(BUILD_EXAMPLES "Compile examples" ON)


# Add the given directories to those the compiler uses to search for include files
include_directories(src)
include_directories(src/apis)
include_directories(src/callbacks)
include_directories(src/hardware/cpu)
include_directories(src/hardware/gpu)
include_directories(src/initializers)
include_directories(src/layers)
include_directories(src/layers/conv)
include_directories(src/layers/core)
include_directories(src/layers/merge)
include_directories(src/layers/noise)
include_directories(src/layers/operators)
include_directories(src/layers/pool)
include_directories(src/layers/recurrent)
include_directories(src/losses)
include_directories(src/metrics)
include_directories(src/optimizers)
include_directories(src/tensor)


# SET SOURCES *******************************
SET(SOURCE_DIR "src")
SET(SOURCES
        src/apis/eddl.cpp
        src/apis/eddl.h
        src/callbacks/callbacks.h
        src/hardware/cpu/cpu_convol.cpp
        src/hardware/cpu/cpu_convol.h
        src/hardware/gpu/tensor_cuda.h
        src/hardware/gpu/tensor_cuda_op.h
        src/hardware/gpu/tensor_kernels.h
        src/initializers/initializer.cpp
        src/initializers/initializer.h
        src/initializers/initializer_constant.cpp
        src/initializers/initializer_glorot_normal.cpp
        src/initializers/initializer_glorot_uniform.cpp
        src/initializers/initializer_indentity.cpp
        src/initializers/initializer_orthogonal.cpp
        src/initializers/initializer_random_normal.cpp
        src/initializers/initializer_random_uniform.cpp
        src/layers/conv/layer_conv.cpp
        src/layers/conv/layer_conv.h
        src/layers/conv/layer_transposed_conv.cpp
        src/layers/conv/layer_upsampling.cpp
        src/layers/core/layer_activation.cpp
        src/layers/core/layer_batchnorm.cpp
        src/layers/core/layer_core.h
        src/layers/core/layer_dense.cpp
        src/layers/core/layer_drop.cpp
        src/layers/core/layer_embedding.cpp
        src/layers/core/layer_input.cpp
        src/layers/core/layer_reshape.cpp
        src/layers/core/layer_tensor.cpp
        src/layers/core/layer_transpose.cpp
        src/layers/merge/layer_add.cpp
        src/layers/merge/layer_average.cpp
        src/layers/merge/layer_concat.cpp
        src/layers/merge/layer_matmul.cpp
        src/layers/merge/layer_maximum.cpp
        src/layers/merge/layer_merge.h
        src/layers/merge/layer_minimum.cpp
        src/layers/merge/layer_subtract.cpp
        src/layers/noise/layer_gaussian.cpp
        src/layers/noise/layer_noise.h
        src/layers/operators/layer_abs.cpp
        src/layers/operators/layer_diff.cpp
        src/layers/operators/layer_div.cpp
        src/layers/operators/layer_exp.cpp
        src/layers/operators/layer_log.cpp
        src/layers/operators/layer_log10.cpp
        src/layers/operators/layer_log2.cpp
        src/layers/operators/layer_mean.cpp
        src/layers/operators/layer_mult.cpp
        src/layers/operators/layer_operator.cpp
        src/layers/operators/layer_operators.h
        src/layers/operators/layer_pow.cpp
        src/layers/operators/layer_sqrt.cpp
        src/layers/operators/layer_sum.cpp
        src/layers/operators/layer_var.cpp
        src/layers/pool/layer_avgpool.cpp
        src/layers/pool/layer_globalavgpool.cpp
        src/layers/pool/layer_globalmaxpool.cpp
        src/layers/pool/layer_maxpool.cpp
        src/layers/pool/layer_pool.cpp
        src/layers/pool/layer_pool.h
        src/layers/recurrent/layer_lstm.cpp
        src/layers/recurrent/layer_recurrent.h
        src/layers/recurrent/layer_rnn.cpp
        src/layers/layer.cpp
        src/layers/layer.h
        src/losses/loss.cpp
        src/losses/loss.h
        src/losses/loss_cross_entropy.cpp
        src/losses/loss_mean_squared_error.cpp
        src/losses/loss_soft_cross_entropy.cpp
        src/metrics/metric.cpp
        src/metrics/metric.h
        src/metrics/metric_categorical_acc.cpp
        src/metrics/metric_mean_squared_error.cpp
        src/optimizers/optim.cpp
        src/optimizers/optim.h
        src/optimizers/optim_adadelta.cpp
        src/optimizers/optim_adagrad.cpp
        src/optimizers/optim_adam.cpp
        src/optimizers/optim_adamax.cpp
        src/optimizers/optim_nadam.cpp
        src/optimizers/optim_rmsprop.cpp
        src/optimizers/optim_sgd.cpp
        src/tensor/tensor.cpp
        src/tensor/tensor.h
        src/tensor/tensor_binary.cpp
        src/tensor/tensor_operators.cpp
        src/tensor/tensor_unary.cpp
        src/tensor/tensor_zero.cpp
        src/compserv.cpp
        src/compserv.h
        src/net.cpp
        src/net.h
        src/utils.cpp
        src/utils.h
        )


# SET TESTS ****************************
SET(TEST_DIR "tests")
SET(TESTS
    "${TEST_DIR}/test_tensor.cpp"
    )


# [LIBRARY] EDDLL LIBRARY *********************
if(EDDLL_SHARED)
	add_library(eddll SHARED "")
else(EDDLL_SHARED)
	add_library(eddll STATIC "")
endif(EDDLL_SHARED)
target_sources(eddll PRIVATE ${SOURCES})
target_include_directories(eddll PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
	$<INSTALL_INTERFACE:include>
)
target_include_directories(eddll PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen>
	$<INSTALL_INTERFACE:include/third_party/eigen>
)
set_target_properties(eddll PROPERTIES OUTPUT_NAME_DEBUG "eddlld")
set_target_properties(eddll PROPERTIES OUTPUT_NAME_RELEASE "eddll")
set_target_properties(eddll PROPERTIES POSITION_INDEPENDENT_CODE ON)

# pthreads setup for windows os
# The following commands should be removed when pthread linux
# specific calls will be replaced by thread c++ standard calls
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
	set(EDDLL_PTHREADS_INSTALL_PATH "$ENV{PTHREADS_ROOT}" CACHE PATH "Path to the installation of pthreads under Windows")
	if(EDDLL_PTHREADS_INSTALL_PATH STREQUAL "")
		message(ERROR " pthreds install path must be specified under Windows")
	endif()
	target_link_libraries(eddll debug "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVC3d.lib"
								debug "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3d.lib"
								debug "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3d.lib")
	target_link_libraries(eddll optimized "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVC3.lib"
								optimized "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3.lib"
								optimized "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3.lib")
	target_include_directories(eddll PUBLIC "${EDDLL_PTHREADS_INSTALL_PATH}/include")
endif()

# pthreads setup for linux os
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    SET(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    SET(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
    target_link_libraries(eddll Threads::Threads)
endif()

# [BINDING] Create python binding
if(BUILD_PYTHON)
    eddll_update_third_party(pybind11)
    set(CMAKE_FOLDER 3party/pybind11)
    add_subdirectory(third_party/pybind11)
    set(CMAKE_FOLDER "")
    SET(C_LIB_NAME "_C")
    pybind11_add_module(${C_LIB_NAME} ${SOURCES} "${SOURCE_DIR}/binding.cpp")
    # can we just link the library instead of adding all the sources and linking pthreads again?
    # target_link_libraries(${C_LIB_NAME} PRIVATE eddll)
    if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
        target_link_libraries(${C_LIB_NAME} PRIVATE "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVC3.lib")
        target_link_libraries(${C_LIB_NAME} PRIVATE "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3.lib")
        target_link_libraries(${C_LIB_NAME} PRIVATE "${EDDLL_PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3.lib")
        target_include_directories(${C_LIB_NAME} PUBLIC "${EDDLL_PTHREADS_INSTALL_PATH}/include")
    endif()
    if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        target_link_libraries(${C_LIB_NAME} PRIVATE Threads::Threads)
    endif()
endif(BUILD_PYTHON)
#unset(BUILD_PYTHON CACHE)

# [TESTS] Generate a test executable
if(BUILD_TESTS)
    eddll_update_third_party(googletest)
    set(CMAKE_FOLDER 3party/googletest)
    add_subdirectory(third_party/googletest)
    SET(UNITTEST_NAME "${PROJECT_NAME}_tests")
    set(CMAKE_FOLDER "")
    add_executable(${UNITTEST_NAME} ${TESTS})
    target_include_directories(${UNITTEST_NAME} PUBLIC ${gtest_SOURCE_DIR}/include)
    target_link_libraries(${UNITTEST_NAME} eddll gtest gtest_main)
    add_test(UnitTests ${UNITTEST_NAME})  # To do "make test"
    enable_testing()
endif(BUILD_TESTS)
#unset(BUILD_TESTS CACHE)


# [EXAMPLES] Generate a few test executables
if(BUILD_EXAMPLES)
    SET(EXAMPLES_DIR "examples")
    set(CMAKE_FOLDER examples)
    add_executable(example_conv "${EXAMPLES_DIR}/eddl_conv.cpp")
    add_executable(example_mlp "${EXAMPLES_DIR}/eddl_mlp.cpp")
  	add_executable(example_predict "${EXAMPLES_DIR}/eddl_predict.cpp")
    target_link_libraries(example_conv eddll)
    target_link_libraries(example_mlp eddll)
  	target_link_libraries(example_predict eddll)
    set(CMAKE_FOLDER "")
endif(BUILD_EXAMPLES)
#unset(BUILD_EXAMPLES CACHE)


# [INSTALL] Install the eddll library
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "....")
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "..." FORCE)
endif()
install(TARGETS eddll
		EXPORT EDDLLConfig
		RUNTIME DESTINATION bin
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib/static
		)
install(EXPORT EDDLLConfig DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/
		DESTINATION include/eddll
        FILES_MATCHING PATTERN "*.h")

set(EIGEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/Eigen")
set(ESCAPED_CMAKE_CURRENT_SOURCE_DIR "${EIGEN_SOURCE_DIR}")
file(GLOB Eigen_directory_files "${EIGEN_SOURCE_DIR}/*")
#message(STATUS ${Eigen_directory_files})
foreach(f ${Eigen_directory_files})
  if(NOT f MATCHES "\\.txt" AND NOT f MATCHES "${ESCAPED_CMAKE_CURRENT_SOURCE_DIR}/[.].+" AND NOT f MATCHES "${ESCAPED_CMAKE_CURRENT_SOURCE_DIR}/src")
    list(APPEND Eigen_directory_files_to_install ${f})
  endif()
endforeach(f ${Eigen_directory_files})

install(FILES
  ${Eigen_directory_files_to_install}
  DESTINATION include/third_party/eigen/Eigen
)
