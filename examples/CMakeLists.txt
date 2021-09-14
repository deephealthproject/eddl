cmake_minimum_required(VERSION 3.9.2)

project(eddl-examples)


# EXAMPLES: MNIST ****************************************************
add_executable(mnist_mlp "nn/1_mnist/1_mnist_mlp.cpp")
target_link_libraries(mnist_mlp eddl)
# distributed
add_executable(mnist_mlp_distr "nn/1_mnist/1_mnist_mlp_distr.cpp")
target_link_libraries(mnist_mlp_distr eddl)

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
# distributed
add_executable(mnist_conv_distr "nn/1_mnist/7_mnist_conv_distr.cpp")
target_link_libraries(mnist_conv_distr eddl)

add_executable(mnist_rnn "nn/1_mnist/8_mnist_rnn.cpp")
target_link_libraries(mnist_rnn eddl)

add_executable(mnist_mlp_func "nn/1_mnist/9_mnist_mlp_func.cpp")
target_link_libraries(mnist_mlp_func eddl)

add_executable(mnist_rnn_func "nn/1_mnist/10_mnist_rnn_func.cpp")
target_link_libraries(mnist_rnn_func eddl)

add_executable(mnist_conv1D "nn/1_mnist/11_mnist_conv1D.cpp")
target_link_libraries(mnist_conv1D eddl)
#distributed
add_executable(mnist_conv1D_distr "nn/1_mnist/11_mnist_conv1D_distr.cpp")
target_link_libraries(mnist_conv1D_distr eddl)

add_executable(mnist_conv_dice "nn/1_mnist/13_mnist_conv_dice.cpp")
target_link_libraries(mnist_conv_dice eddl)

add_executable(mnist_losses "nn/1_mnist/14_mnist_losses.cpp")
target_link_libraries(mnist_losses eddl)



# EXAMPLES: CIFAR10 ****************************************************
add_executable(cifar_conv "nn/2_cifar10/1_cifar_conv.cpp")
target_link_libraries(cifar_conv eddl)
#distributed
add_executable(cifar_conv_distr "nn/2_cifar10/1_cifar_conv_distr.cpp")
target_link_libraries(cifar_conv_distr eddl)

add_executable(cifar_conv_da "nn/2_cifar10/2_cifar_conv_da.cpp")
target_link_libraries(cifar_conv_da eddl)

add_executable(cifar_vgg16 "nn/2_cifar10/3_cifar_vgg16.cpp")
target_link_libraries(cifar_vgg16 eddl)
#distributed
add_executable(cifar_vgg16_distr "nn/2_cifar10/3_cifar_vgg16_distr.cpp")
target_link_libraries(cifar_vgg16_distr eddl)

add_executable(cifar_vgg16_bn "nn/2_cifar10/4_cifar_vgg16_bn.cpp")
target_link_libraries(cifar_vgg16_bn eddl)
#distributed
add_executable(cifar_vgg16_bn_distr "nn/2_cifar10/4_cifar_vgg16_bn_distr.cpp")
target_link_libraries(cifar_vgg16_bn_distr eddl)

add_executable(cifar_vgg16_gn "nn/2_cifar10/4_cifar_vgg16_gn.cpp")
target_link_libraries(cifar_vgg16_gn eddl)
#distributed
add_executable(cifar_vgg16_gn_distr "nn/2_cifar10/4_cifar_vgg16_gn_distr.cpp")
target_link_libraries(cifar_vgg16_gn_distr eddl)

add_executable(cifar_resnet "nn/2_cifar10/5_cifar_resnet.cpp")
target_link_libraries(cifar_resnet eddl)
#distributed
add_executable(cifar_resnet_distr "nn/2_cifar10/5_cifar_resnet_distr.cpp")
target_link_libraries(cifar_resnet_distr eddl)

add_executable(cifar_resnet_da_bn "nn/2_cifar10/6_cifar_resnet_da_bn.cpp")
target_link_libraries(cifar_resnet_da_bn eddl)

add_executable(cifar_resnet50_da_bn "nn/2_cifar10/7_cifar_resnet50_da_bn.cpp")
target_link_libraries(cifar_resnet50_da_bn eddl)
#distributed
add_executable(cifar_resnet50_da_bn_distr "nn/2_cifar10/7_cifar_resnet50_da_bn_distr.cpp")
target_link_libraries(cifar_resnet50_da_bn_distr eddl)

add_executable(cifar_avgpool3d "nn/2_cifar10/8_cifar_avgpool3d.cpp")
target_link_libraries(cifar_avgpool3d eddl)

# EXAMPLES: Drive segmentation ****************************************************
add_executable(drive_seg "nn/3_drive/1_drive_seg.cpp")
target_link_libraries(drive_seg eddl)

# EXAMPLES: NLP ****************************************************

add_executable(nlp_sentiment_rnn "nn/4_NLP/1_nlp_sentiment_rnn.cpp")
target_link_libraries(nlp_sentiment_rnn eddl)

add_executable(nlp_sentiment_lstm "nn/4_NLP/2_nlp_sentiment_lstm.cpp")
target_link_libraries(nlp_sentiment_lstm eddl)

add_executable(nlp_sentiment_gru "nn/4_NLP/2_nlp_sentiment_gru.cpp")
target_link_libraries(nlp_sentiment_gru eddl)

add_executable(nlp_machine_translation "nn/4_NLP/3_nlp_machine_translation.cpp")
target_link_libraries(nlp_machine_translation eddl)

add_executable(nlp_video_to_labels "nn/4_NLP/4_nlp_video_to_labels.cpp")
target_link_libraries(nlp_video_to_labels eddl)

add_executable(nlp_text_generation "nn/4_NLP/5_nlp_text_generation.cpp")
target_link_libraries(nlp_text_generation eddl)

# EXAMPLES: SYNTHETIC IMAGENET ****************************************************
add_executable(synthetic_imagenet_vgg16 "nn/5_synthetic_imagenet/1_imagenet_vgg16.cpp")
target_link_libraries(synthetic_imagenet_vgg16 eddl)
add_executable(synthetic_imagenet_vgg16_bn "nn/5_synthetic_imagenet/2_imagenet_vgg16_bn.cpp")
target_link_libraries(synthetic_imagenet_vgg16_bn eddl)

# TEST INTERNALS ****************************************************
  add_executable(test1 "test_internals/test1.cpp")
  target_link_libraries(test1 eddl)
  add_executable(test2 "test_internals/test2.cpp")
  target_link_libraries(test2 eddl)
  add_executable(test3 "test_internals/test3.cpp")
  target_link_libraries(test3 eddl)
  add_executable(test4 "test_internals/test4.cpp")
  target_link_libraries(test4 eddl)

if(BUILD_PROTOBUF)
    add_executable(test_onnx_auxiliar "test_internals/onnx/test_onnx_auxiliar_layers.cpp")
    target_link_libraries(test_onnx_auxiliar eddl)

	add_executable(test_onnx_conv1D "test_internals/onnx/test_onnx_conv1D.cpp")
    target_link_libraries(test_onnx_conv1D eddl)
    add_executable(test_onnx_conv2D "test_internals/onnx/test_onnx_conv2D.cpp")
    target_link_libraries(test_onnx_conv2D eddl)
    add_executable(test_onnx_conv3D "test_internals/onnx/test_onnx_conv3D.cpp")
    target_link_libraries(test_onnx_conv3D eddl)

	add_executable(test_onnx_dilated_conv1D "test_internals/onnx/test_onnx_dilated_conv1D.cpp")
	target_link_libraries(test_onnx_dilated_conv1D eddl)
	add_executable(test_onnx_dilated_conv2D "test_internals/onnx/test_onnx_dilated_conv2D.cpp")
	target_link_libraries(test_onnx_dilated_conv2D eddl)
	add_executable(test_onnx_dilated_conv3D "test_internals/onnx/test_onnx_dilated_conv3D.cpp")
	target_link_libraries(test_onnx_dilated_conv3D eddl)

	add_executable(test_onnx_convT2D "test_internals/onnx/test_onnx_convT2D.cpp")
    target_link_libraries(test_onnx_convT2D eddl)
    add_executable(test_onnx_convT3D "test_internals/onnx/test_onnx_convT3D.cpp")
    target_link_libraries(test_onnx_convT3D eddl)

	add_executable(test_onnx_upsample2D "test_internals/onnx/test_onnx_upsample2D.cpp")
	target_link_libraries(test_onnx_upsample2D eddl)
	add_executable(test_onnx_upsample3D "test_internals/onnx/test_onnx_upsample3D.cpp")
	target_link_libraries(test_onnx_upsample3D eddl)

	add_executable(test_onnx_lstm_imdb "test_internals/onnx/test_onnx_lstm_imdb.cpp")
    target_link_libraries(test_onnx_lstm_imdb eddl)
    add_executable(test_onnx_gru_imdb "test_internals/onnx/test_onnx_gru_imdb.cpp")
    target_link_libraries(test_onnx_gru_imdb eddl)

    add_executable(test_onnx_lstm_mnist "test_internals/onnx/test_onnx_lstm_mnist.cpp")
    target_link_libraries(test_onnx_lstm_mnist eddl)
    add_executable(test_onnx_gru_mnist "test_internals/onnx/test_onnx_gru_mnist.cpp")
    target_link_libraries(test_onnx_gru_mnist eddl)
    add_executable(test_onnx_rnn_mnist "test_internals/onnx/test_onnx_rnn_mnist.cpp")
    target_link_libraries(test_onnx_rnn_mnist eddl)

    add_executable(test_onnx_lstm_enc_dec "test_internals/onnx/test_onnx_lstm_enc_dec.cpp")
    target_link_libraries(test_onnx_lstm_enc_dec eddl)
    add_executable(test_onnx_gru_enc_dec "test_internals/onnx/test_onnx_gru_enc_dec.cpp")
    target_link_libraries(test_onnx_gru_enc_dec eddl)
endif()

# EXAMPLES: Tensor ****************************************************
add_executable(tensor_ops "tensor/eddl_ops.cpp")
target_link_libraries(tensor_ops eddl)

add_executable(tensor_image "tensor/eddl_image.cpp")
target_link_libraries(tensor_image eddl)

add_executable(tensor_io "tensor/eddl_io.cpp")
target_link_libraries(tensor_io eddl)

add_executable(tensor_dev_tests "tensor/eddl_dev_tests.cpp")
target_link_libraries(tensor_dev_tests eddl)

# EXAMPLES: ONNX ******************************************************************
if(BUILD_PROTOBUF)
    add_executable(onnx_pointer "onnx/1_onnx_pointer.cpp")
    target_link_libraries(onnx_pointer eddl)

    add_executable(onnx_export "onnx/2_onnx_save_net_to_file.cpp")
    target_link_libraries(onnx_export eddl)

    add_executable(onnx_import "onnx/3_onnx_import_net_from_file.cpp")
    target_link_libraries(onnx_import eddl)

    add_executable(onnx_utils_serialization "onnx/4_onnx_utils_serialization.cpp")
    target_link_libraries(onnx_utils_serialization eddl)

    add_executable(onnx_gradients "onnx/4_onnx_test_gradients.cpp")
    target_link_libraries(onnx_gradients eddl)

    add_executable(onnx_import_reshape "onnx/5_onnx_import_net_and_reshape.cpp")
    target_link_libraries(onnx_import_reshape eddl)

	add_executable(onnx_predict "onnx/6_onnx_predict.cpp")
	target_link_libraries(onnx_predict eddl)

	# EXAMPLES: ONNX MNIST ****************************************************
	add_executable(onnx_mnist_mlp "onnx/nn/1_mnist/1_mnist_mlp.cpp")
	target_link_libraries(onnx_mnist_mlp eddl)

	add_executable(onnx_mnist_mlp_initializers "onnx/nn/1_mnist/2_mnist_mlp_initializers.cpp")
	target_link_libraries(onnx_mnist_mlp_initializers eddl)

	add_executable(onnx_mnist_mlp_regularizers "onnx/nn/1_mnist/3_mnist_mlp_regularizers.cpp")
	target_link_libraries(onnx_mnist_mlp_regularizers eddl)

	add_executable(onnx_mnist_mlp_da "onnx/nn/1_mnist/4_mnist_mlp_da.cpp")
	target_link_libraries(onnx_mnist_mlp_da eddl)

	add_executable(onnx_mnist_mlp_train_batch "onnx/nn/1_mnist/5_mnist_mlp_train_batch.cpp")
	target_link_libraries(onnx_mnist_mlp_train_batch eddl)

	add_executable(onnx_mnist_auto_encoder "onnx/nn/1_mnist/6_mnist_auto_encoder.cpp")
	target_link_libraries(onnx_mnist_auto_encoder eddl)

	add_executable(onnx_mnist_conv "onnx/nn/1_mnist/7_mnist_conv.cpp")
	target_link_libraries(onnx_mnist_conv eddl)

	add_executable(onnx_mnist_rnn "onnx/nn/1_mnist/8_mnist_rnn.cpp")
	target_link_libraries(onnx_mnist_rnn eddl)

	add_executable(onnx_mnist_mlp_func "onnx/nn/1_mnist/9_mnist_mlp_func.cpp")
	target_link_libraries(onnx_mnist_mlp_func eddl)

	add_executable(onnx_mnist_rnn_func "onnx/nn/1_mnist/10_mnist_rnn_func.cpp")
	target_link_libraries(onnx_mnist_rnn_func eddl)

	add_executable(onnx_mnist_conv1D "onnx/nn/1_mnist/11_mnist_conv1D.cpp")
	target_link_libraries(onnx_mnist_conv1D eddl)


	add_executable(onnx_mnist_conv_dice "onnx/nn/1_mnist/13_mnist_conv_dice.cpp")
	target_link_libraries(onnx_mnist_conv_dice eddl)

	add_executable(onnx_mnist_losses "onnx/nn/1_mnist/14_mnist_losses.cpp")
	target_link_libraries(onnx_mnist_losses eddl)



	# EXAMPLES: ONNX CIFAR10 ****************************************************
	add_executable(onnx_cifar_conv "onnx/nn/2_cifar10/1_cifar_conv.cpp")
	target_link_libraries(onnx_cifar_conv eddl)

	add_executable(onnx_cifar_conv_da "onnx/nn/2_cifar10/2_cifar_conv_da.cpp")
	target_link_libraries(onnx_cifar_conv_da eddl)

	add_executable(onnx_cifar_vgg16 "onnx/nn/2_cifar10/3_cifar_vgg16.cpp")
	target_link_libraries(onnx_cifar_vgg16 eddl)

	add_executable(onnx_cifar_vgg16_bn "onnx/nn/2_cifar10/4_cifar_vgg16_bn.cpp")
	target_link_libraries(onnx_cifar_vgg16_bn eddl)

	add_executable(onnx_cifar_vgg16_gn "onnx/nn/2_cifar10/4_cifar_vgg16_gn.cpp")
	target_link_libraries(onnx_cifar_vgg16_gn eddl)

	add_executable(onnx_cifar_resnet "onnx/nn/2_cifar10/5_cifar_resnet.cpp")
	target_link_libraries(onnx_cifar_resnet eddl)

	add_executable(onnx_cifar_resnet_da_bn "onnx/nn/2_cifar10/6_cifar_resnet_da_bn.cpp")
	target_link_libraries(onnx_cifar_resnet_da_bn eddl)

	add_executable(onnx_cifar_resnet50_da_bn "onnx/nn/2_cifar10/7_cifar_resnet50_da_bn.cpp")
	target_link_libraries(onnx_cifar_resnet50_da_bn eddl)



# EXAMPLES: Drive segmentation ****************************************************
	add_executable(onnx_drive_seg "onnx/nn/3_drive/1_drive_seg.cpp")
	target_link_libraries(onnx_drive_seg eddl)


# EXAMPLES: ONNX NLP ****************************************************

	add_executable(onnx_nlp_sentiment_rnn "onnx/nn/4_NLP/1_nlp_sentiment_rnn.cpp")
	target_link_libraries(onnx_nlp_sentiment_rnn eddl)

	add_executable(onnx_nlp_sentiment_lstm "onnx/nn/4_NLP/2_nlp_sentiment_lstm.cpp")
	target_link_libraries(onnx_nlp_sentiment_lstm eddl)

	add_executable(onnx_nlp_machine_translation "onnx/nn/4_NLP/3_nlp_machine_translation.cpp")
	target_link_libraries(onnx_nlp_machine_translation eddl)

endif()
