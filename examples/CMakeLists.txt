# EXAMPLES: MNIST ****************************************************
add_executable(mnist_mlp "${EXAMPLES_DIR}/nn/1_mnist/1_mnist_mlp.cpp")
target_link_libraries(mnist_mlp eddl)

add_executable(mnist_mlp_initializers "${EXAMPLES_DIR}/nn/1_mnist/2_mnist_mlp_initializers.cpp")
target_link_libraries(mnist_mlp_initializers eddl)

add_executable(mnist_mlp_regularizers "${EXAMPLES_DIR}/nn/1_mnist/3_mnist_mlp_regularizers.cpp")
target_link_libraries(mnist_mlp_regularizers eddl)

add_executable(mnist_mlp_da "${EXAMPLES_DIR}/nn/1_mnist/4_mnist_mlp_da.cpp")
target_link_libraries(mnist_mlp_da eddl)

add_executable(mnist_mlp_train_batch "${EXAMPLES_DIR}/nn/1_mnist/5_mnist_mlp_train_batch.cpp")
target_link_libraries(mnist_mlp_train_batch eddl)

add_executable(mnist_auto_encoder "${EXAMPLES_DIR}/nn/1_mnist/6_mnist_auto_encoder.cpp")
target_link_libraries(mnist_auto_encoder eddl)

add_executable(mnist_conv "${EXAMPLES_DIR}/nn/1_mnist/7_mnist_conv.cpp")
target_link_libraries(mnist_conv eddl)

# EXAMPLES: CIFAR10 ****************************************************
add_executable(cifar_conv "${EXAMPLES_DIR}/nn/2_cifar10/1_cifar_conv.cpp")
target_link_libraries(cifar_conv eddl)

add_executable(cifar_conv_da "${EXAMPLES_DIR}/nn/2_cifar10/2_cifar_conv_da.cpp")
target_link_libraries(cifar_conv_da eddl)

add_executable(cifar_vgg16 "${EXAMPLES_DIR}/nn/2_cifar10/3_cifar_vgg16.cpp")
target_link_libraries(cifar_vgg16 eddl)

add_executable(cifar_vgg16_bn "${EXAMPLES_DIR}/nn/2_cifar10/4_cifar_vgg16_bn.cpp")
target_link_libraries(cifar_vgg16_bn eddl)

add_executable(cifar_vgg16_gn "${EXAMPLES_DIR}/nn/2_cifar10/4_cifar_vgg16_gn.cpp")
target_link_libraries(cifar_vgg16_gn eddl)

add_executable(cifar_resnet "${EXAMPLES_DIR}/nn/2_cifar10/5_cifar_resnet.cpp")
target_link_libraries(cifar_resnet eddl)

add_executable(cifar_resnet_da_bn "${EXAMPLES_DIR}/nn/2_cifar10/6_cifar_resnet_da_bn.cpp")
target_link_libraries(cifar_resnet_da_bn eddl)

add_executable(cifar_resnet50 "${EXAMPLES_DIR}/nn/2_cifar10/7_cifar_resnet50_da_bn.cpp")
target_link_libraries(cifar_resnet50 eddl)


# EXAMPLES: Drive segmentation ****************************************************
add_executable(drive_seg "${EXAMPLES_DIR}/nn/3_drive/1_drive_seg.cpp")
target_link_libraries(drive_seg eddl)


# EXAMPLES: Tensor ****************************************************
add_executable(tensor_ops "${EXAMPLES_DIR}/tensor/eddl_ops.cpp")
target_link_libraries(tensor_ops eddl)

add_executable(tensor_image "${EXAMPLES_DIR}/tensor/eddl_image.cpp")
target_link_libraries(tensor_image eddl)

add_executable(tensor_io "${EXAMPLES_DIR}/tensor/eddl_io.cpp")
target_link_libraries(tensor_io eddl)

add_executable(tensor_tests_dev "${EXAMPLES_DIR}/tensor/eddl_tests_dev.cpp")
target_link_libraries(tensor_tests_dev eddl)


# EXAMPLES: ONNX ******************************************************************
if(BUILD_PROTOBUF)
    add_executable(onnx_pointer "${EXAMPLES_DIR}/onnx/1_onnx_pointer.cpp")
    target_link_libraries(onnx_pointer eddl)

    add_executable(onnx_export "${EXAMPLES_DIR}/onnx/2_onnx_save_net_to_file.cpp")
    target_link_libraries(onnx_export eddl)

    add_executable(onnx_import "${EXAMPLES_DIR}/onnx/3_onnx_import_net_from_file.cpp")
    target_link_libraries(onnx_import eddl)
endif()