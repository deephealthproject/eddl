#!/usr/bin/env bash

#
# This script is devoted to find memory leaks, double free of memory blocks
# and accesses to non-valid memory addresses.
#
# This script assumes SANITIZERS are set in the call to CMAKE, either for
# "Debug" and "Release" CMAKE_BUILD_TYPE
#
# As can be seen below, all the examples are executed with two parameters:
#
#  --testing :: to use a small number of training epochs and a subset of the correspondingdataset
#
#  --cpu :: to use CPU computing service, when SANITIZERS are set GPU code fails
#
#
# Please, check which examples are included in this script, new examples must be
# added manually and must be ready to accept the above mentioned parameters in
# order to be used with this scripts, otherwise the execution of not correctly prepared
# examples will fail or last too much time.
#
#

current_dir=$(pwd)
parent_dir=$(dirname ${current_dir})
this_dir=$(basename ${current_dir})

if [ ${this_dir} != "build" ]
then
    echo "****"
    echo "**** You are not located in the proper directory to run these tests!"
    echo "****"
    echo "**** BYE"
    echo "****"
    exit 1
fi

cd bin
current_dir=$(pwd)
this_dir=${current_dir##${parent_dir}/}

if [ ${this_dir} != "build/bin" ]
then
    echo "****"
    echo "**** You are not located in the proper directory to run these tests!"
    echo "****"
    echo "**** BYE"
    echo "****"
    exit 1
fi

echo "**********************************************************"
echo "******************** RUNNING EXAMPLES ********************"
echo "**********************************************************"


mnist_executables=""
mnist_executables="${mnist_executables} mnist_auto_encoder"
mnist_executables="${mnist_executables} mnist_conv"
mnist_executables="${mnist_executables} mnist_conv1D"
mnist_executables="${mnist_executables} mnist_conv_dice"
mnist_executables="${mnist_executables} mnist_losses"
mnist_executables="${mnist_executables} mnist_mlp"
mnist_executables="${mnist_executables} mnist_mlp_da"
mnist_executables="${mnist_executables} mnist_mlp_func"
mnist_executables="${mnist_executables} mnist_mlp_initializers"
mnist_executables="${mnist_executables} mnist_mlp_regularizers"
mnist_executables="${mnist_executables} mnist_mlp_train_batch"
mnist_executables="${mnist_executables} mnist_rnn"
mnist_executables="${mnist_executables} mnist_rnn_func"

nlp_executables=""
nlp_executables="${nlp_executables} nlp_machine_translation"
nlp_executables="${nlp_executables} nlp_sentiment_gru"
nlp_executables="${nlp_executables} nlp_sentiment_lstm"
nlp_executables="${nlp_executables} nlp_sentiment_rnn"
nlp_executables="${nlp_executables} nlp_text_generation"
nlp_executables="${nlp_executables} nlp_video_to_labels"

cifar10_executables=""
cifar10_executables="${cifar10_executables} cifar_conv"
cifar10_executables="${cifar10_executables} cifar_conv_da"
cifar10_executables="${cifar10_executables} cifar_resnet"
cifar10_executables="${cifar10_executables} cifar_resnet50_da_bn"
cifar10_executables="${cifar10_executables} cifar_resnet_da_bn"
cifar10_executables="${cifar10_executables} cifar_vgg16"
cifar10_executables="${cifar10_executables} cifar_vgg16_bn"
cifar10_executables="${cifar10_executables} cifar_vgg16_gn"

segmentation_executables=""
segmentation_executables="${segmentation_executables} drive_seg"

synthetic_imagenet_executables=""
synthetic_imagenet_executables="${synthetic_imagenet_executables} synthetic_imagenet_vgg16"
synthetic_imagenet_executables="${synthetic_imagenet_executables} synthetic_imagenet_vgg16_bn"

onnx_executables=""
onnx_executables="${onnx_executables} onnx_pointer"
#onnx_executables="${onnx_executables} onnx_export"
#onnx_executables="${onnx_executables} onnx_import"
#onnx_executables="${onnx_executables} onnx_import_reshape"
#onnx_executables="${onnx_executables} onnx_gradients"
#onnx_executables="${onnx_executables} onnx_cifar_conv"
#onnx_executables="${onnx_executables} onnx_cifar_conv_da"
#onnx_executables="${onnx_executables} onnx_cifar_resnet"
#onnx_executables="${onnx_executables} onnx_cifar_resnet50_da_bn"
#onnx_executables="${onnx_executables} onnx_cifar_resnet_da_bn"
#onnx_executables="${onnx_executables} onnx_cifar_vgg16"
#onnx_executables="${onnx_executables} onnx_cifar_vgg16_bn"
#onnx_executables="${onnx_executables} onnx_cifar_vgg16_gn"
#onnx_executables="${onnx_executables} onnx_mnist_auto_encoder"
#onnx_executables="${onnx_executables} onnx_mnist_conv"
#onnx_executables="${onnx_executables} onnx_mnist_conv1D"
#onnx_executables="${onnx_executables} onnx_mnist_conv_dice"
#onnx_executables="${onnx_executables} onnx_mnist_losses"
#onnx_executables="${onnx_executables} onnx_mnist_mlp"
#onnx_executables="${onnx_executables} onnx_mnist_mlp_da"
#onnx_executables="${onnx_executables} onnx_mnist_mlp_func"
#onnx_executables="${onnx_executables} onnx_mnist_mlp_initializers"
#onnx_executables="${onnx_executables} onnx_mnist_mlp_regularizers"
#onnx_executables="${onnx_executables} onnx_mnist_mlp_train_batch"
#onnx_executables="${onnx_executables} onnx_mnist_rnn"
#onnx_executables="${onnx_executables} onnx_mnist_rnn_func"
#onnx_executables="${onnx_executables} onnx_nlp_machine_translation"
#onnx_executables="${onnx_executables} onnx_nlp_sentiment_lstm"
#onnx_executables="${onnx_executables} onnx_nlp_sentiment_rnn"
#onnx_executables="${onnx_executables} onnx_drive_seg"

parameters="--testing --cpu"

for executable in   ${mnist_executables} \
                    ${cifar10_executables} \
                    ${nlp_executables} \
                    ${segmentation_executables} \
                    ${synthetic_imagenet_executables} \
                    ${onnx_executables}
do
    echo "###################################################################"
    echo "###################################################################"
    echo "######"
    echo "######    ${executable} ${parameters}"
    echo "######"
    echo "###################################################################"
    echo "###################################################################"
    ./${executable} ${parameters}
    echo "*******************************************************************"
    echo "*******************************************************************"
    echo ""
    echo ""
    echo ""
done 2>&1 | tee ../tests-sanitizers.output

echo ""
echo ""
echo "Please, look at file tests-sanitizers.output"
echo ""
echo ""

exit 0
