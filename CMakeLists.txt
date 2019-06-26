cmake_minimum_required(VERSION 3.9.2)
PROJECT(eddl)

# SET C++ COMPILER STANDARD
SET(CMAKE_CXX_STANDARD 11)

# SET COMPILER FLAGS
#set(CMAKE_CXX_FLAGS "-march=native -mavx -O3")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE} )
message(STATUS "C++ flags: " ${CMAKE_CXX_FLAGS} )

# SET THREADS *******************************
SET(CMAKE_THREAD_PREFER_PTHREAD TRUE)
SET(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# SET GTEST *******************************
#find_package(GTest REQUIRED)
#include_directories(${GTEST_INCLUDE_DIRS})
add_subdirectory(third_party/googletest)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# SET LIBRARIES *******************************
add_subdirectory(third_party/pybind11)
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

# [TESTS] ****************************
SET(TEST_DIR "tests")
SET(TESTS
    "${TEST_DIR}/test_tensor.cpp"
    )


# [BINDING] Create python binding
if(BUILD_PYTHON)
    SET(C_LIB_NAME "_C")
    pybind11_add_module(${C_LIB_NAME} ${SOURCES} "${SOURCE_DIR}/binding.cpp")
endif(BUILD_PYTHON)
unset(BUILD_PYTHON CACHE)

# [TESTS] Generate a test executable
if(BUILD_TESTS)
    SET(UNITTEST_NAME "${PROJECT_NAME}_tests")
    add_executable(${UNITTEST_NAME} ${SOURCES} ${TESTS})
    target_link_libraries(${UNITTEST_NAME} gtest gtest_main Threads::Threads)
    add_test(UnitTests ${UNITTEST_NAME})  # To do "make test"
endif(BUILD_TESTS)
unset(BUILD_TESTS CACHE)

## [EXAMPLES] Generate a test executable
    if(BUILD_EXAMPLES)
    SET(EXAMPLES_DIR "examples")
    add_executable(example_conv ${SOURCES} "${EXAMPLES_DIR}/eddl_conv.cpp")
    add_executable(example_mlp ${SOURCES} "${EXAMPLES_DIR}/eddl_mlp.cpp")
    add_executable(example_predict ${SOURCES} "${EXAMPLES_DIR}/eddl_predict.cpp")
    target_link_libraries(example_conv Threads::Threads)
    target_link_libraries(example_mlp Threads::Threads)
    target_link_libraries(example_predict Threads::Threads)
endif(BUILD_EXAMPLES)
unset(BUILD_EXAMPLES CACHE)
