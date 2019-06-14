cmake_minimum_required(VERSION 3.9.2)
PROJECT(eddl VERSION 0.0.1 DESCRIPTION "Shared library for EDDL")

# SET C++ COMPILER STANDARD
SET(CMAKE_CXX_STANDARD 11)

# SET VARIABLES
#SET(SO_VERSION 1)

# SET THREADS *******************************
SET(CMAKE_THREAD_PREFER_PTHREAD TRUE)
SET(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)


# Add the given directories to those the compiler uses to search for include files
include_directories(src/cc/eddl)
include_directories(src/cc/eddl/apis)
include_directories(src/cc/eddl/callbacks)
include_directories(src/cc/eddl/hardware/cpu)
include_directories(src/cc/eddl/hardware/gpu)
include_directories(src/cc/eddl/initializers)
include_directories(src/cc/eddl/layers)
include_directories(src/cc/eddl/layers/conv)
include_directories(src/cc/eddl/layers/core)
include_directories(src/cc/eddl/layers/merge)
include_directories(src/cc/eddl/layers/noise)
include_directories(src/cc/eddl/layers/operators)
include_directories(src/cc/eddl/layers/pool)
include_directories(src/cc/eddl/layers/recurrent)
include_directories(src/cc/eddl/losses)
include_directories(src/cc/eddl/metrics)
include_directories(src/cc/eddl/optimizers)
include_directories(src/cc/eddl/tensor)


# SET SOURCES *******************************
SET(SOURCE_DIR "src/cc/eddl")
SET(SOURCES
        src/cc/eddl/apis/eddl.cpp
        src/cc/eddl/apis/eddl.h
        src/cc/eddl/callbacks/callbacks.h
        src/cc/eddl/hardware/cpu/cpu_convol.cpp
        src/cc/eddl/hardware/cpu/cpu_convol.h
        src/cc/eddl/hardware/gpu/tensor_cuda.h
        src/cc/eddl/hardware/gpu/tensor_cuda_op.h
        src/cc/eddl/hardware/gpu/tensor_kernels.h
        src/cc/eddl/initializers/initializer.cpp
        src/cc/eddl/initializers/initializer.h
        src/cc/eddl/initializers/initializer_constant.cpp
        src/cc/eddl/initializers/initializer_glorot_normal.cpp
        src/cc/eddl/initializers/initializer_glorot_uniform.cpp
        src/cc/eddl/initializers/initializer_indentity.cpp
        src/cc/eddl/initializers/initializer_orthogonal.cpp
        src/cc/eddl/initializers/initializer_random_normal.cpp
        src/cc/eddl/initializers/initializer_random_uniform.cpp
        src/cc/eddl/layers/conv/layer_conv.cpp
        src/cc/eddl/layers/conv/layer_conv.h
        src/cc/eddl/layers/conv/layer_transposed_conv.cpp
        src/cc/eddl/layers/conv/layer_upsampling.cpp
        src/cc/eddl/layers/core/layer_activation.cpp
        src/cc/eddl/layers/core/layer_batchnorm.cpp
        src/cc/eddl/layers/core/layer_core.h
        src/cc/eddl/layers/core/layer_dense.cpp
        src/cc/eddl/layers/core/layer_drop.cpp
        src/cc/eddl/layers/core/layer_embedding.cpp
        src/cc/eddl/layers/core/layer_input.cpp
        src/cc/eddl/layers/core/layer_reshape.cpp
        src/cc/eddl/layers/core/layer_tensor.cpp
        src/cc/eddl/layers/core/layer_transpose.cpp
        src/cc/eddl/layers/merge/layer_add.cpp
        src/cc/eddl/layers/merge/layer_average.cpp
        src/cc/eddl/layers/merge/layer_concat.cpp
        src/cc/eddl/layers/merge/layer_matmul.cpp
        src/cc/eddl/layers/merge/layer_maximum.cpp
        src/cc/eddl/layers/merge/layer_merge.h
        src/cc/eddl/layers/merge/layer_minimum.cpp
        src/cc/eddl/layers/merge/layer_subtract.cpp
        src/cc/eddl/layers/noise/layer_gaussian.cpp
        src/cc/eddl/layers/noise/layer_noise.h
        src/cc/eddl/layers/operators/layer_abs.cpp
        src/cc/eddl/layers/operators/layer_diff.cpp
        src/cc/eddl/layers/operators/layer_div.cpp
        src/cc/eddl/layers/operators/layer_exp.cpp
        src/cc/eddl/layers/operators/layer_log.cpp
        src/cc/eddl/layers/operators/layer_log10.cpp
        src/cc/eddl/layers/operators/layer_log2.cpp
        src/cc/eddl/layers/operators/layer_mean.cpp
        src/cc/eddl/layers/operators/layer_mult.cpp
        src/cc/eddl/layers/operators/layer_operator.cpp
        src/cc/eddl/layers/operators/layer_operators.h
        src/cc/eddl/layers/operators/layer_pow.cpp
        src/cc/eddl/layers/operators/layer_sqrt.cpp
        src/cc/eddl/layers/operators/layer_sum.cpp
        src/cc/eddl/layers/operators/layer_var.cpp
        src/cc/eddl/layers/pool/layer_avgpool.cpp
        src/cc/eddl/layers/pool/layer_globalavgpool.cpp
        src/cc/eddl/layers/pool/layer_globalmaxpool.cpp
        src/cc/eddl/layers/pool/layer_maxpool.cpp
        src/cc/eddl/layers/pool/layer_pool.cpp
        src/cc/eddl/layers/pool/layer_pool.h
        src/cc/eddl/layers/recurrent/layer_lstm.cpp
        src/cc/eddl/layers/recurrent/layer_recurrent.h
        src/cc/eddl/layers/recurrent/layer_rnn.cpp
        src/cc/eddl/layers/layer.cpp
        src/cc/eddl/layers/layer.h
        src/cc/eddl/losses/loss.cpp
        src/cc/eddl/losses/loss.h
        src/cc/eddl/losses/loss_cross_entropy.cpp
        src/cc/eddl/losses/loss_mean_squared_error.cpp
        src/cc/eddl/losses/loss_soft_cross_entropy.cpp
        src/cc/eddl/metrics/metric.cpp
        src/cc/eddl/metrics/metric.h
        src/cc/eddl/metrics/metric_categorical_acc.cpp
        src/cc/eddl/metrics/metric_mean_squared_error.cpp
        src/cc/eddl/optimizers/optim.cpp
        src/cc/eddl/optimizers/optim.h
        src/cc/eddl/optimizers/optim_adadelta.cpp
        src/cc/eddl/optimizers/optim_adagrad.cpp
        src/cc/eddl/optimizers/optim_adam.cpp
        src/cc/eddl/optimizers/optim_adamax.cpp
        src/cc/eddl/optimizers/optim_nadam.cpp
        src/cc/eddl/optimizers/optim_rmsprop.cpp
        src/cc/eddl/optimizers/optim_sgd.cpp
        src/cc/eddl/tensor/tensor.cpp
        src/cc/eddl/tensor/tensor.h
        src/cc/eddl/tensor/tensor_binary.cpp
        src/cc/eddl/tensor/tensor_operators.cpp
        src/cc/eddl/tensor/tensor_unary.cpp
        src/cc/eddl/tensor/tensor_zero.cpp
        src/cc/eddl/compserv.cpp
        src/cc/eddl/compserv.h
        src/cc/eddl/net.cpp
        src/cc/eddl/net.h
        src/cc/eddl/utils.cpp
        src/cc/eddl/utils.h
        )

# [TESTS] ****************************
SET(TEST_DIR "tests/cc")
SET(TESTS
    "${TEST_DIR}/test_main.cpp"
    "${TEST_DIR}/test_tensor.cpp"
    )

# [EXAMPLES] ****************************
#SET(EXAMPLES_DIR "examples/cc")
#SET(EXAMPLES
#        "${EXAMPLES_DIR}/eddl_mlp.cpp"
#        #"${EXAMPLES_DIR}/eddl_conv.cpp"
#        )

# [SHARED LIBRARY]
##include_directories(${CMAKE_SOURCE_DIR}/pyeddl)
##link_directories(${CMAKE_SOURCE_DIR}/pyeddl)


# Create shared library
#include(GNUInstallDirs)
##add_library(${PROJECT_NAME} SHARED ${SOURCES})
#set_target_properties(${PROJECT_NAME} PROPERTIES
#        VERSION ${PROJECT_VERSION}
#        SOVERSION ${SO_VERSION}
#        PUBLIC_HEADER src/cc/eddl/apis/wrapper.h)
#configure_file(${PROJECT_NAME}.pc.in ${PROJECT_NAME}.pc @ONLY)

# Link thread library
#target_include_directories(${PROJECT_NAME} PRIVATE src/cc/eddl)
##target_link_libraries(${PROJECT_NAME} Threads::Threads)

## Install shared library
#install(TARGETS ${PROJECT_NAME}
#        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc
#        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
#add_executable(${PROJECT_NAME} ${SOURCES})

# [TESTS] Generate a test executable
SET(TEST_LIB_NAME "${PROJECT_NAME}_test")
include_directories(third_party/eigen)
add_subdirectory(third_party/catch2)
add_executable(${TEST_LIB_NAME} ${SOURCES} ${TESTS})
target_link_libraries(${TEST_LIB_NAME} Catch2::Catch2)
target_link_libraries(${TEST_LIB_NAME} Threads::Threads)

# [BINDING] Create pyhton binding
SET(C_LIB_NAME "_C")
add_subdirectory(third_party/pybind11)
pybind11_add_module(${C_LIB_NAME} ${SOURCES} "${SOURCE_DIR}/binding.cpp")

