cmake_minimum_required(VERSION 3.9.2)
PROJECT(eddl LANGUAGES CXX)

# SET C++ COMPILER STANDARD
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# USER OPTIONS *******************************
option(BUILD_PYTHON "Compile python binding" OFF)
option(BUILD_TESTS "Compile tests" OFF)
option(BUILD_EXAMPLES "Compile examples" ON)
SET(BUILD_TARGET "cpu" CACHE STRING "Compile library for {cpu, gpu, fpga}")

# PROPERTIES *******************************
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")


# SET BUILD TYPE (RELEASE/DEBUG)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# SET COMPILER FLAGS
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    #set(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}") // This is a bad idea
    set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# CUDA FLAGS
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -lcudart -lcublas -lcurand")
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(EDDL_WITH_CUDA OFF CACHE BOOL "Enable CUDA support if checked")

# Device specific setup
if(BUILD_TARGET STREQUAL "cpu")
    message("Compiling for CPU...")
elseif(BUILD_TARGET STREQUAL "gpu")
    message("Compiling for GPU...")
    set(EDDL_WITH_CUDA ON)
elseif(BUILD_TARGET STREQUAL "fpga")
    message("Compiling for FPGA...")
    message( FATAL_ERROR "FPGA not implemented, CMake will exit." )
endif()

# SET STATIC/DYNAMIC RUNTIME
set(EDDL_SHARED OFF CACHE BOOL "Whether to build a static or dynamic library")

# SHOW OPTIONS
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE} )
message(STATUS "Build target: " ${BUILD_TARGET} )
message(STATUS "Build tests: " ${BUILD_TESTS} )
message(STATUS "Build examples: " ${BUILD_EXAMPLES} )
message(STATUS "Build python: " ${BUILD_PYTHON} )
message(STATUS "CUDA Enabled: " ${EDDL_WITH_CUDA} )
message(STATUS "C++ flags: " ${CMAKE_CXX_FLAGS} )
message(STATUS "CUDA flags: " ${CMAKE_CUDA_FLAGS} )

# Add cuda (if possible)
if(EDDL_WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(WARNING "No CUDA support")
    endif()
endif()

# Microsoft Visual C++ - set static/dynamic runtime
if(MSVC)
    if(EDDL_SHARED)
        message(STATUS "Setting dinamic runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
        if (EDDL_WITH_CUDA AND CMAKE_CUDA_COMPILER)
			foreach(flag_var CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE CMAKE_CUDA_FLAGS_MINSIZEREL CMAKE_CUDA_FLAGS_RELWITHDEBINFO) 
				string(REGEX REPLACE "-MT" "-MD" ${flag_var} "${${flag_var}}")
			endforeach(flag_var)
        endif()
    else(EDDL_SHARED)
        message(STATUS "Setting static runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
        if (EDDL_WITH_CUDA AND CMAKE_CUDA_COMPILER)
			foreach(flag_var CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE CMAKE_CUDA_FLAGS_MINSIZEREL CMAKE_CUDA_FLAGS_RELWITHDEBINFO) 
				string(REGEX REPLACE "-MD" "-MT" ${flag_var} "${${flag_var}}")
			endforeach(flag_var)
        endif()
	endif(EDDL_SHARED)

	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()


# [MACRO] Download submodules
macro(eddl_update_third_party SUBMODULE)
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
eddl_update_third_party(eigen)
include_directories(third_party/eigen)


# Add the given directories to those the compiler uses to search for include files
include_directories(src)
include_directories(src/apis)
include_directories(src/callbacks)
include_directories(src/hardware/cpu)
include_directories(src/hardware/cpu/nn)
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
include_directories(src/descriptors)


# SET SOURCES *******************************
SET(SOURCE_DIR "src")
SET(CPP_SOURCES
        src/apis/eddl.cpp
        src/apis/eddl.h
        src/callbacks/callbacks.h
        src/hardware/cpu/cpu_hw.h
        src/hardware/cpu/cpu_comparison.cpp
        src/hardware/cpu/cpu_core.cpp
        src/hardware/cpu/cpu_generator.cpp
        src/hardware/cpu/cpu_math.cpp
        src/hardware/cpu/cpu_reduction.cpp
        src/hardware/cpu/nn/cpu_activations.cpp
        src/hardware/cpu/nn/cpu_conv.cpp
        src/hardware/cpu/nn/cpu_losses.cpp
        src/hardware/cpu/nn/cpu_metrics.cpp
        src/hardware/cpu/nn/cpu_pool.cpp
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
        src/layers/generators/layer_generator.cpp
        src/layers/generators/layer_generators.h
        src/layers/generators/layer_gauss.cpp
        src/layers/generators/layer_uniform.cpp
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
        src/layers/operators/layer_mult.cpp
        src/layers/operators/layer_operator.cpp
        src/layers/operators/layer_operators.h
        src/layers/operators/layer_pow.cpp
        src/layers/operators/layer_sqrt.cpp
        src/layers/operators/layer_sum.cpp
        src/layers/pool/layer_avgpool.cpp
        src/layers/pool/layer_globalavgpool.cpp
        src/layers/pool/layer_globalmaxpool.cpp
        src/layers/pool/layer_maxpool.cpp
        src/layers/pool/layer_pool.cpp
        src/layers/pool/layer_pool.h
        src/layers/recurrent/layer_lstm.cpp
        src/layers/recurrent/layer_recurrent.h
        src/layers/recurrent/layer_rnn.cpp
        src/layers/reductions/layer_mean.cpp
        src/layers/reductions/layer_reduction.cpp
        src/layers/reductions/layer_reductions.h
        src/layers/reductions/layer_var.cpp
        src/layers/reductions/layer_sum.cpp
        src/layers/reductions/layer_max.cpp
        src/layers/reductions/layer_min.cpp
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
        src/descriptors/descriptors.h
        src/descriptors/descriptor_conv.cpp
        src/descriptors/descriptor_pool.cpp
        src/tensor/tensor.h
        src/tensor/tensor.cpp
        src/tensor/tensor_comparison.cpp
        src/tensor/tensor_core.cpp
        src/tensor/tensor_create.cpp
        src/tensor/tensor_generator.cpp
        src/tensor/tensor_math.cpp
        src/tensor/tensor_reduction.cpp
        src/tensor/tensor_serialization.cpp
        src/tensor/nn/tensor_nn.h
        src/tensor/nn/tensor_activations.cpp
        src/tensor/nn/tensor_conv.cpp
        src/tensor/nn/tensor_losses.cpp
        src/tensor/nn/tensor_metrics.cpp
        src/tensor/nn/tensor_pool.cpp
        src/compserv.cpp
        src/compserv.h
        src/net.cpp
        src/net.h
        src/system_info.h
        src/utils.cpp
        src/utils.h
        src/random.cpp
        src/random.h
        )
SET(CUDA_SOURCES
        src/hardware/gpu/tensor_cuda.cu
        src/hardware/gpu/tensor_cuda.h
        src/hardware/gpu/tensor_cuda_op.cu
        src/hardware/gpu/tensor_cuda_op.h
        src/hardware/gpu/tensor_kernels.cu
        src/hardware/gpu/tensor_kernels.h
        )
SET(SOURCES ${CPP_SOURCES} ${CUDA_SOURCES})

# SET TESTS ****************************
SET(TEST_DIR "tests")
SET(TESTS
    "${TEST_DIR}/test_tensor.cpp"
    )


# [LIBRARY] EDDL LIBRARY *********************
if(EDDL_SHARED)
	add_library(eddl SHARED "")
else(EDDL_SHARED)
	add_library(eddl STATIC "")
endif(EDDL_SHARED)
target_sources(eddl PRIVATE ${CPP_SOURCES})
if(EDDL_WITH_CUDA AND CMAKE_CUDA_COMPILER)
    target_sources(eddl PRIVATE ${CUDA_SOURCES})
endif()
target_include_directories(eddl PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
	$<INSTALL_INTERFACE:include>
)
target_include_directories(eddl PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen>
	$<INSTALL_INTERFACE:include/third_party/eigen>
)
#target_compile_features(eddl PUBLIC cxx_std_11)


# Target properties
set_target_properties(eddl PROPERTIES OUTPUT_NAME_DEBUG "eddld")
set_target_properties(eddl PROPERTIES OUTPUT_NAME_RELEASE "eddl")
set_target_properties(eddl PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(eddl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# pthreads setup for windows os
# The following commands should be removed when pthread linux
# specific calls will be replaced by thread c++ standard calls
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
	set(EDDL_PTHREADS_INSTALL_PATH "$ENV{PTHREADS_ROOT}" CACHE PATH "Path to the installation of pthreads under Windows")
	if(EDDL_PTHREADS_INSTALL_PATH STREQUAL "")
		message(ERROR " pthreds install path must be specified under Windows")
	endif()
	target_link_libraries(eddl debug "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVC3d.lib"
								debug "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3d.lib"
								debug "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3d.lib")
	target_link_libraries(eddl optimized "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVC3.lib"
								optimized "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3.lib"
								optimized "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3.lib")
	target_include_directories(eddl PUBLIC "${EDDL_PTHREADS_INSTALL_PATH}/include")
endif()

# pthreads setup for linux os
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    SET(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    SET(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
    target_link_libraries(eddl Threads::Threads)
endif()

## [BINDING] Create python binding
#if(BUILD_PYTHON)
#    eddl_update_third_party(pybind11)
#    set(CMAKE_FOLDER 3party/pybind11)
#    add_subdirectory(third_party/pybind11)
#    set(CMAKE_FOLDER "")
#    SET(C_LIB_NAME "_C")
#    pybind11_add_module(${C_LIB_NAME} ${SOURCES} "${SOURCE_DIR}/binding.cpp")
#    # can we just link the library instead of adding all the sources and linking pthreads again?
#    # target_link_libraries(${C_LIB_NAME} PRIVATE eddl)
#    if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
#        target_link_libraries(${C_LIB_NAME} PRIVATE "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVC3.lib")
#        target_link_libraries(${C_LIB_NAME} PRIVATE "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVCE3.lib")
#        target_link_libraries(${C_LIB_NAME} PRIVATE "${EDDL_PTHREADS_INSTALL_PATH}/lib/libpthreadVSE3.lib")
#        target_include_directories(${C_LIB_NAME} PUBLIC "${EDDL_PTHREADS_INSTALL_PATH}/include")
#    endif()
#    if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
#        target_link_libraries(${C_LIB_NAME} PRIVATE Threads::Threads)
#    endif()
#endif(BUILD_PYTHON)
##unset(BUILD_PYTHON CACHE)

# [TESTS] Generate a test executable
if(BUILD_TESTS)
    eddl_update_third_party(googletest)
    set(CMAKE_FOLDER 3party/googletest)
    add_subdirectory(third_party/googletest)
    SET(UNITTEST_NAME "${PROJECT_NAME}_tests")
    set(CMAKE_FOLDER "")
    add_executable(${UNITTEST_NAME} ${TESTS})
    target_include_directories(${UNITTEST_NAME} PUBLIC ${gtest_SOURCE_DIR}/include)
    target_link_libraries(${UNITTEST_NAME} eddl gtest gtest_main)
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
    add_executable(example_ae "${EXAMPLES_DIR}/eddl_ae.cpp")
    add_executable(example_dae "${EXAMPLES_DIR}/eddl_dae.cpp")
    add_executable(example_dae_class "${EXAMPLES_DIR}/eddl_dae_class.cpp")
    add_executable(example_test "${EXAMPLES_DIR}/eddl_test.cpp")
    add_executable(example_load_save "${EXAMPLES_DIR}/eddl_load_save.cpp")
    add_executable(example_gpu "${EXAMPLES_DIR}/eddl_gpu.cpp")
    add_executable(example_temp "${EXAMPLES_DIR}/eddl_temp.cpp")
    add_executable(example_tensor "${EXAMPLES_DIR}/eddl_tensor.cpp")
    target_link_libraries(example_conv eddl)
    target_link_libraries(example_mlp eddl)
  	target_link_libraries(example_predict eddl)
    target_link_libraries(example_ae eddl)
    target_link_libraries(example_dae eddl)
    target_link_libraries(example_dae_class eddl)
    target_link_libraries(example_test eddl)
    target_link_libraries(example_load_save eddl)
    target_link_libraries(example_gpu eddl)
    target_link_libraries(example_tensor eddl)
    target_link_libraries(example_temp eddl)
    set(CMAKE_FOLDER "")
endif(BUILD_EXAMPLES)
#unset(BUILD_EXAMPLES CACHE)


# [INSTALL] Install the EDDL library
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "..." FORCE)
endif()
install(TARGETS eddl
		EXPORT EDDLTargets
		RUNTIME DESTINATION bin
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
install(EXPORT EDDLTargets DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/
		DESTINATION include/eddl
        FILES_MATCHING PATTERN "*.h")

# [INSTALL] Install Eigen files
include("${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/cmake/RegexUtils.cmake")
test_escape_string_as_regex()
escape_string_as_regex(ESCAPED_EIGEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/Eigen")
file(GLOB Eigen_directory_files "${ESCAPED_EIGEN_SOURCE_DIR}/*")
#message(STATUS ${Eigen_directory_files})
foreach(f ${Eigen_directory_files})
    if(NOT f MATCHES "\\.txt" AND NOT f MATCHES "${ESCAPED_EIGEN_SOURCE_DIR}/[.].+" AND NOT f MATCHES "${ESCAPED_EIGEN_SOURCE_DIR}/src")
        list(APPEND Eigen_directory_files_to_install ${f})
    endif()
endforeach(f ${Eigen_directory_files})
install(FILES
        ${Eigen_directory_files_to_install}
        DESTINATION "include/third_party/eigen/Eigen"
        )
install(DIRECTORY ${ESCAPED_EIGEN_SOURCE_DIR}/src DESTINATION "include/third_party/eigen/Eigen" COMPONENT Devel FILES_MATCHING PATTERN "*.h")


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EDDLConfig.cmake.in ${CMAKE_BINARY_DIR}/cmake/EDDLConfig.cmake @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/cmake/EDDLConfig.cmake
        DESTINATION ${CMAKE_INSTALL_PREFIX}) 
