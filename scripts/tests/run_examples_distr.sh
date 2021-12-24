#!/usr/bin/env bash

echo "**********************************************************"
echo "******************** RUNNING EXAMPLES ********************"
echo "**********************************************************"

BUILD_TARGET=$1
EPOCHS=$2
$BUILD_PATH=build
MPI="mpirun -np 4 -hostfile ../cluster.altec -map-by node"

$MPI ./$BUILD_PATH/bin/cifar_conv_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/cifar_conv_da_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/cifar_resnet_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/cifar_resnet50_da_bn_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/cifar_resnet_da_bn_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/cifar_vgg16_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/cifar_vgg16_bn_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/cifar_vgg16_gn_distr $BUILD_TARGET $EPOCHS --min-acc 0.3

$MPI ./$BUILD_PATH/bin/drive_seg $BUILD_TARGET $EPOCHS --min-acc 0.3

$MPI ./$BUILD_PATH/bin/mnist_auto_encoder_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/mnist_conv_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/mnist_conv1D_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/mnist_conv_dice_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/mnist_losses_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/mnist_mlp_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/mnist_mlp_da_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/mnist_mlp_func_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/mnist_mlp_initializers_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/mnist_mlp_regularizers_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/mnist_mlp_train_batch_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/mnist_rnn_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/mnist_rnn_func_distr $BUILD_TARGET $EPOCHS --min-acc 0.3

$MPI ./$BUILD_PATH/bin/nlp_machine_translation_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/nlp_sentiment_gru_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/nlp_sentiment_lstm_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/nlp_sentiment_rnn_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/nlp_text_generation_distr $BUILD_TARGET $EPOCHS --min-acc 0.3
$MPI ./$BUILD_PATH/bin/nlp_video_to_labels_distr $BUILD_TARGET $EPOCHS --min-acc 0.3

#./$BUILD_PATH/bin/onnx_cifar_conv $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_cifar_conv_da $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_cifar_resnet $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_cifar_resnet50_da_bn $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_cifar_resnet_da_bn $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_cifar_vgg16 $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_cifar_vgg16_bn $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_cifar_vgg16_gn $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_drive_seg $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_export $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_import $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_mnist_auto_encoder $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_mnist_conv $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_mnist_conv1D $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_mnist_conv_dice $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_mnist_losses $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_mnist_mlp $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_mnist_mlp_da $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_mnist_mlp_func $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_mnist_mlp_initializers $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_mnist_mlp_regularizers $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_mnist_mlp_train_batch $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_mnist_rnn $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_mnist_rnn_func $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_nlp_machine_translation $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_nlp_sentiment_lstm $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_nlp_sentiment_rnn $BUILD_TARGET $EPOCHS --min-acc 0.3
#./$BUILD_PATH/bin/onnx_pointer $BUILD_TARGET $EPOCHS --min-acc 0.3