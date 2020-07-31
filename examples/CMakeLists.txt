cmake_minimum_required(VERSION 3.9.2)

project(eddl-examples)


# EXAMPLES: MNIST ****************************************************
add_executable(mnist_mlp "nn/1_mnist/1_mnist_mlp.cpp")
target_link_libraries(mnist_mlp eddl)

add_executable(mnist_mlp_initializers "nn/1_mnist/2_mnist_mlp_initializers.cpp")
target_link_libraries(mnist_mlp_initializers eddl)

add_executable(mnist_mlp_regularizers "nn/1_mnist/3_mnist_mlp_regularizers.cpp")
target_link_libraries(mnist_mlp_regularizers eddl)

add_executable(mnist_mlp_da "nn/1_mnist/4_mnist_mlp_da.cpp")
target_link_libraries(mnist_mlp_da eddl)

add_executable(mnist_mlp_train_batch "nn/1_mnist/5_mnist_mlp_train_batch.cpp")
target_link_libraries(mnist_mlp_train_batch eddl)

add_executable(mnist_auto_encoder "nn/1_mnist/6_mnist_auto_encoder.cpp")
target_link_libraries(mnist_auto_encoder eddl)

add_executable(mnist_conv "nn/1_mnist/7_mnist_conv.cpp")
target_link_libraries(mnist_conv eddl)

add_executable(mnist_rnn "nn/1_mnist/8_mnist_rnn.cpp")
target_link_libraries(mnist_rnn eddl)

add_executable(mnist_mlp_func "nn/1_mnist/9_mnist_mlp_func.cpp")
target_link_libraries(mnist_mlp_func eddl)

add_executable(mnist_rnn_func "nn/1_mnist/10_mnist_rnn_func.cpp")
target_link_libraries(mnist_rnn_func eddl)

add_executable(mnist_conv1D "nn/1_mnist/11_mnist_conv1D.cpp")
target_link_libraries(mnist_conv1D eddl)

add_executable(mnist_auto_merging "nn/1_mnist/12_mnist_auto_encoder_merging.cpp")
target_link_libraries(mnist_auto_merging eddl)

add_executable(mnist_conv_dice "nn/1_mnist/13_mnist_conv_dice.cpp")
target_link_libraries(mnist_conv_dice eddl)

add_executable(mnist_losses "nn/1_mnist/14_mnist_losses.cpp")
target_link_libraries(mnist_losses eddl)

add_executable(mnist_siamese "nn/1_mnist/15_mnist_siamese.cpp")
target_link_libraries(mnist_siamese eddl)


# EXAMPLES: CIFAR10 ****************************************************
add_executable(cifar_conv "nn/2_cifar10/1_cifar_conv.cpp")
target_link_libraries(cifar_conv eddl)

add_executable(cifar_conv_da "nn/2_cifar10/2_cifar_conv_da.cpp")
target_link_libraries(cifar_conv_da eddl)

add_executable(cifar_vgg16 "nn/2_cifar10/3_cifar_vgg16.cpp")
target_link_libraries(cifar_vgg16 eddl)

add_executable(cifar_vgg16_bn "nn/2_cifar10/4_cifar_vgg16_bn.cpp")
target_link_libraries(cifar_vgg16_bn eddl)

add_executable(cifar_vgg16_gn "nn/2_cifar10/4_cifar_vgg16_gn.cpp")
target_link_libraries(cifar_vgg16_gn eddl)

add_executable(cifar_resnet "nn/2_cifar10/5_cifar_resnet.cpp")
target_link_libraries(cifar_resnet eddl)

add_executable(cifar_resnet_da_bn "nn/2_cifar10/6_cifar_resnet_da_bn.cpp")
target_link_libraries(cifar_resnet_da_bn eddl)

add_executable(cifar_resnet50_da_bn "nn/2_cifar10/7_cifar_resnet50_da_bn.cpp")
target_link_libraries(cifar_resnet50_da_bn eddl)


# EXAMPLES: Drive segmentation ****************************************************
add_executable(drive_seg "nn/3_drive/1_drive_seg.cpp")
target_link_libraries(drive_seg eddl)

# EXAMPLES: NLP ****************************************************

add_executable(nlp_sentiment_rnn "nn/4_NLP/1_nlp_sentiment_rnn.cpp")
target_link_libraries(nlp_sentiment_rnn eddl)

add_executable(nlp_sentiment_lstm "nn/4_NLP/2_nlp_sentiment_lstm.cpp")
target_link_libraries(nlp_sentiment_lstm eddl)

add_executable(nlp_machine_translation "nn/4_NLP/3_nlp_machine_translation.cpp")
target_link_libraries(nlp_machine_translation eddl)

add_executable(nlp_text_generation "nn/4_NLP/4_nlp_text_generation.cpp")
target_link_libraries(nlp_text_generation eddl)


# EXAMPLES: Tensor ****************************************************
add_executable(tensor_ops "tensor/eddl_ops.cpp")
target_link_libraries(tensor_ops eddl)

add_executable(tensor_image "tensor/eddl_image.cpp")
target_link_libraries(tensor_image eddl)

add_executable(tensor_io "tensor/eddl_io.cpp")
target_link_libraries(tensor_io eddl)

add_executable(tensor_tests_dev "tensor/eddl_tests_dev.cpp")
target_link_libraries(tensor_tests_dev eddl)


# EXAMPLES: ONNX ******************************************************************
if(BUILD_PROTOBUF)
    add_executable(onnx_pointer "onnx/1_onnx_pointer.cpp")
    target_link_libraries(onnx_pointer eddl)

    add_executable(onnx_export "onnx/2_onnx_save_net_to_file.cpp")
    target_link_libraries(onnx_export eddl)

    add_executable(onnx_import "onnx/3_onnx_import_net_from_file.cpp")
    target_link_libraries(onnx_import eddl)

    add_executable(onnx_gradients "onnx/4_onnx_test_gradients.cpp")
    target_link_libraries(onnx_gradients eddl)
endif()
