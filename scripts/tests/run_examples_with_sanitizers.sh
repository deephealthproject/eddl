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

parameters="--testing --cpu"

for executable in ${mnist_executables} ${nlp_executables} # nlp_* cifar_* onnx_* drive_*
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
