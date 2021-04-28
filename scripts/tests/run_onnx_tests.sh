#!/usr/bin/env bash

# This script runs all the ONNX tests to check if the export/import
# of ONNX models works properly. Runs import/exports tests between
# EDDL computing services and with other libraries (Keras, Pytorch,
# ONNX Runtime). The idea is that after training, exporting, importing
# and evaluating a model the loss/metric obtained must be the same
# before and after the export/import.
#
# IMPORTANT:
#   - This script must be executed with the EDDL compiled and
#     from the "build" directory created for compilation.
#
#   - To run the python tests you need to have some python dependecies
#     installed: tensorflow, keras, torch (pytorch), torchvision, torchtext,
#     onnx, onnxruntime, onnx2keras, keras2onnx, onnx-simplifier, tqdm.
#
#   - In case of using a python environment to manage the dependecies
#     remember to activate it before running the script.
#

#################### HELPER FUNCTIONS
print_help() {
    echo "Script Arguments:"
    echo "  -p PATH  Path to the folder with python test scripts"
    echo "  -e  Only execute EDDL export -> EDDL import tests"
    echo "  -d  To execute the EDDL scripts just with the default device"
    echo "  -h  To show this help message =)"
}

print_header() {
    echo "###################################################################"
    echo "##"
    echo "##  $*"
    echo "##"
    echo "###################################################################"
}

handle_exit_status() {
    if [ "$?" = "0" ]; then
        echo "$1 => OK" >> $2
    else
        echo "$1 => FAIL" >> $2
    fi
}
####################

# Check if we are in the correct folder
if [[ ! -d "bin" ]]
then
    echo "*****************************************************************"
    echo "Directory \"bin\" not found! You must execute this script"
    echo "from the \"build\" directory and with the EDDL compiled."
    echo "*****************************************************************"
    exit 1
fi

build_dir=$(pwd)

# Scripts folders
eddl_bin="${build_dir}/bin"
py_scripts="../scripts/tests/py_onnx"  # Relative to build directory

# Output logs
scripts_output_filename="tests_onnx_outputs.out"
tests_results_filename="tests_onnx_results.out"
scripts_output_path="${build_dir}/${scripts_output_filename}"
tests_results_path="${build_dir}/${tests_results_filename}"

# Execution flags
only_eddl=0  # To execute just the EDDL to EDDL tests

# Parse script flags
while getopts p:eh flag
do
    case "${flag}" in
        p)
            py_scripts=${OPTARG}
            if [[ ! -d $output_path ]]; then
                echo "The python test scripts path provided is not valid!"
                exit 1
            fi
            ;;
        e)
            only_eddl=1
            ;;
        h) 
            print_help
            exit 0
            ;;
        \?)
            # Invalid argument detected
            print_help
            exit 1
            ;;
    esac
done

# Go to bin folder to execute EDDL scripts
pushd ${eddl_bin} > /dev/null

# Store each test as a triplet separated by ";" -> "[TEST_NAME_TO_SHOW];[ONNX_FILE_NAME];[BIN_FILE_NAME],[POSSIBLE_FLAGS,]"
#   Note: The second element can be a list separated by "," to add possible script arguments
scripts_to_run=()
scripts_to_run+=("EDDL_to_EDDL_conv1D;test_onnx_conv1D;test_onnx_conv1D")
scripts_to_run+=("EDDL_to_EDDL_conv2D;test_onnx_conv2D;test_onnx_conv2D")
scripts_to_run+=("EDDL_to_EDDL_conv3D;test_onnx_conv3D;test_onnx_conv3D")
scripts_to_run+=("EDDL_to_EDDL_convT2D;test_onnx_convT2D;test_onnx_convT2D")
scripts_to_run+=("EDDL_to_EDDL_GRU_imdb;test_onnx_gru_imdb;test_onnx_gru_imdb")
scripts_to_run+=("EDDL_to_EDDL_LSTM_imdb;test_onnx_lstm_imdb;test_onnx_lstm_imdb")
scripts_to_run+=("EDDL_to_EDDL_GRU_mnist;test_onnx_gru_mnist;test_onnx_gru_mnist")
scripts_to_run+=("EDDL_to_EDDL_LSTM_mnist;test_onnx_lstm_mnist;test_onnx_lstm_mnist")
scripts_to_run+=("EDDL_to_EDDL_RNN_mnist;test_onnx_rnn_mnist;test_onnx_rnn_mnist")
scripts_to_run+=("EDDL_to_EDDL_LSTM_enc_dec;test_onnx_lstm_enc_dec;test_onnx_lstm_enc_dec")
scripts_to_run+=("EDDL_to_EDDL_GRU_enc_dec;test_onnx_gru_enc_dec;test_onnx_gru_enc_dec")
# From EDDL CPU to EDDL CPU
scripts_to_run+=("EDDL_to_EDDL_conv1D_CPU;test_onnx_conv1D_cpu;test_onnx_conv1D,--cpu")
scripts_to_run+=("EDDL_to_EDDL_conv2D_CPU;test_onnx_conv2D_cpu;test_onnx_conv2D,--cpu")
scripts_to_run+=("EDDL_to_EDDL_conv3D_CPU;test_onnx_conv3D_cpu;test_onnx_conv3D,--cpu")
#scripts_to_run+=("EDDL_to_EDDL_convT2D_CPU;test_onnx_convT2D_cpu;test_onnx_convT2D,--cpu") Convt2D not available in CPU
scripts_to_run+=("EDDL_to_EDDL_GRU_imdb_CPU;test_onnx_gru_imdb_cpu;test_onnx_gru_imdb,--cpu")
scripts_to_run+=("EDDL_to_EDDL_LSTM_imdb_CPU;test_onnx_lstm_imdb_cpu;test_onnx_lstm_imdb,--cpu")
scripts_to_run+=("EDDL_to_EDDL_GRU_mnist_CPU;test_onnx_gru_mnist_cpu;test_onnx_gru_mnist,--cpu")
scripts_to_run+=("EDDL_to_EDDL_LSTM_mnist_CPU;test_onnx_lstm_mnist_cpu;test_onnx_lstm_mnist,--cpu")
scripts_to_run+=("EDDL_to_EDDL_RNN_mnist_CPU;test_onnx_rnn_mnist_cpu;test_onnx_rnn_mnist,--cpu")
scripts_to_run+=("EDDL_to_EDDL_LSTM_enc_dec_CPU;test_onnx_lstm_enc_dec_cpu;test_onnx_lstm_enc_dec,--cpu")
scripts_to_run+=("EDDL_to_EDDL_GRU_enc_dec_CPU;test_onnx_gru_enc_dec_cpu;test_onnx_gru_enc_dec,--cpu")

# Prepare output files
print_header "ONNX TESTS RESULTS" > $tests_results_path
print_header "ONNX TESTS FULL OUTPUT" > $scripts_output_path
echo "###################################################################"
echo "#"
echo "# Writing tests results to $tests_results_filename"
echo "#"
echo "# Writing tests scripts full output to $scripts_output_filename"
echo "#"
echo "###################################################################"

# Run "EDDL -> EDDL" tests and store results
print_header "EDDL export -> EDDL import" >> $tests_results_path
for test_data in "${scripts_to_run[@]}"
do
    # Split the test data to get each element
    IFS=';'; read -r -a test_data_arr <<< "${test_data}"  # Split string by ";"

    test_name="${test_data_arr[0]}"   # Get test name to show

    # Get model name to store ONNX and metrics
    model_name="${test_data_arr[1]}"  
    model_path="${eddl_bin}/model_${model_name}.onnx"
    model_metric="${eddl_bin}/metric_${model_name}.txt"

    script_args="${test_data_arr[2]}" # Get comma separated list or script args

    # Get script file path and args to execute the test
    IFS=','; read -r -a script_argv <<< "${script_args}"  # Split string by ","

    echo "Running $test_name"
    print_header "${script_argv[@]} --onnx-file ${model_path} --target-metric ${model_metric}" >> ${scripts_output_path}
    ./${script_argv[@]} --onnx-file ${model_path} --target-metric ${model_metric} &>> ${scripts_output_path}
    handle_exit_status "$test_name" ${tests_results_path}
    echo ""
done

popd > /dev/null  # Go back to build dir

if [[ $only_eddl -eq 0 ]]
then
    ##################################################################################
    # Going to execute the tests with other libraries (ONNX Runtime, Pytorch, Keras) #
    ##################################################################################

    pushd ${py_scripts} > /dev/null # Enter python scripts folder to execute py tests

    #------------------------------------
    # EDDL export -> ONNX Runtime import
    #------------------------------------

    pushd onnx_runtime > /dev/null  # Enter the onnx_runtime folder to execute the import scripts

    # eddl2onnxrt is a list of triplets to define the tests for exporting from EDDL and importing with ONNX Runtime.
    # Each postion of the triplet is:
    #   0: Experiment name to show.
    #   1: ONNX model name to import (created previously with the EDDL scripts).
    #   2: ONNX Runtime script and arguments tu execute for the test.
    # Note:
    #   - To simulate the list of triplets, the triplets are strings with the elements separated by ";".
    #   - The script names and the arguments to execute them are separated by ",".
    eddl2onnxrt=()
    eddl2onnxrt+=("EDDL_to_ONNXRT_conv1D;test_onnx_conv1D;onnxruntime_mnist.py,--input-1D,--no-channel")
    eddl2onnxrt+=("EDDL_to_ONNXRT_conv2D;test_onnx_conv2D;onnxruntime_mnist.py,--input-1D,--no-channel")
    eddl2onnxrt+=("EDDL_to_ONNXRT_conv3D;test_onnx_conv3D;onnxruntime_conv3d_synthetic.py")
    eddl2onnxrt+=("EDDL_to_ONNXRT_convT2D;test_onnx_convT2D;onnxruntime_enc_dec_mnist.py")
    eddl2onnxrt+=("EDDL_to_ONNXRT_GRU_imdb;test_onnx_gru_imdb;onnxruntime_imdb_keras.py,--unsqueeze-input")
    eddl2onnxrt+=("EDDL_to_ONNXRT_LSTM_imdb;test_onnx_lstm_imdb;onnxruntime_imdb_keras.py,--unsqueeze-input")
    eddl2onnxrt+=("EDDL_to_ONNXRT_LSTM_enc_dec;test_onnx_lstm_enc_dec;onnxruntime_recurrent_enc_dec_mnist.py")
    eddl2onnxrt+=("EDDL_to_ONNXRT_GRU_enc_dec;test_onnx_gru_enc_dec;onnxruntime_recurrent_enc_dec_mnist.py")
    # From EDDL CPU to ONNX RT
    eddl2onnxrt+=("EDDL_to_ONNXRT_conv1D_CPU;test_onnx_conv1D_cpu;onnxruntime_mnist.py,--input-1D,--no-channel")
    eddl2onnxrt+=("EDDL_to_ONNXRT_conv2D_CPU;test_onnx_conv2D_cpu;onnxruntime_mnist.py,--input-1D,--no-channel")
    eddl2onnxrt+=("EDDL_to_ONNXRT_conv3D_CPU;test_onnx_conv3D_cpu;onnxruntime_conv3d_synthetic.py")
    #eddl2onnxrt+=("EDDL_to_ONNXRT_convT2D_CPU;test_onnx_convT2D_cpu;onnxruntime_enc_dec_mnist.py") Convt2D not available in CPU
    eddl2onnxrt+=("EDDL_to_ONNXRT_GRU_imdb_CPU;test_onnx_gru_imdb_cpu;onnxruntime_imdb_keras.py,--unsqueeze-input")
    eddl2onnxrt+=("EDDL_to_ONNXRT_LSTM_imdb_CPU;test_onnx_lstm_imdb_cpu;onnxruntime_imdb_keras.py,--unsqueeze-input")
    eddl2onnxrt+=("EDDL_to_ONNXRT_LSTM_enc_dec_CPU;test_onnx_lstm_enc_dec_cpu;onnxruntime_recurrent_enc_dec_mnist.py")
    eddl2onnxrt+=("EDDL_to_ONNXRT_GRU_enc_dec_CPU;test_onnx_gru_enc_dec_cpu;onnxruntime_recurrent_enc_dec_mnist.py")

    # Run "EDDL export -> ONNX Runtime import" tests and store results
    print_header "EDDL export -> ONNX Runtime import" >> $tests_results_path
    for test_data in "${eddl2onnxrt[@]}"
    do
        IFS=';'; read -r -a test_arr <<< "${test_data}"  # Split string by ";"

        # Get test name to show
        test_name="${test_arr[0]}"

        # Get paths to onnx file and target metric to get in test
        model_name="${test_arr[1]}"
        model_path="${eddl_bin}/model_${model_name}.onnx"
        model_metric="${eddl_bin}/metric_${model_name}.txt"

        # Get script path and arguments to execute it
        script_args="${test_arr[2]}"
        IFS=','; read -r -a script_argv <<< "${script_args}"  # Split string by ","
        script_call="python ${script_argv[@]} --onnx-file ${model_path} --target-metric ${model_metric}"

        # Execute test
        echo "Running $test_name"
        print_header "$script_call" >> ${scripts_output_path}
        python ${script_argv[0]} ${script_argv[@]:1} --onnx-file ${model_path} --target-metric ${model_metric} &>> ${scripts_output_path}
        handle_exit_status "$test_name" ${tests_results_path}
        echo ""
    done

    popd > /dev/null  # Go back to py_scripts main folder

    #-------------------------------
    # EDDL export -> Pytorch import
    #-------------------------------

#    TODO: Prepare ONNX models in order to import them with Pytorch. Pytorch can't find the initializers in the current location.
#    pushd pytorch > /dev/null  # Going to execute Pytorch import scripts
#
#    # eddl2pytorch is a list of triplets to define the tests for exporting from EDDL and importing with Pytorch.
#    # Each postion of the triplet is:
#    #   0: Experiment name to show.
#    #   1: ONNX model name to import (created previously with the EDDL scripts).
#    #   2: Pytorch script and arguments tu execute for the test.
#    # Note:
#    #   - To simulate the list of triplets, the triplets are strings with the elements separated by ";".
#    #   - The script names and the arguments to execute them are separated by ",".
#    eddl2pytorch=()
#    eddl2pytorch+=("EDDL_to_Pytorch_conv1D;test_onnx_conv1D;import_scripts/mnist_pytorch_import.py,--input-1D,--no-channel")
#    eddl2pytorch+=("EDDL_to_Pytorch_conv2D;test_onnx_conv2D;import_scripts/mnist_pytorch_import.py,--input-1D,--no-channel")
#    #eddl2pytorch+=("EDDL_to_Pytorch_conv3D;test_onnx_conv3D;import_scripts/conv3d_synthetic_pytorch_import.py") TODO: Pytorch synthetic 3D import
#    eddl2pytorch+=("EDDL_to_Pytorch_GRU_imdb;test_onnx_gru_imdb;import_scripts/imdb_pytorch_import.py,--unsqueeze-input")
#    eddl2pytorch+=("EDDL_to_Pytorch_LSTM_imdb;test_onnx_lstm_imdb;import_scripts/imdb_pytorch_import.py,--unsqueeze-input")
#    eddl2pytorch+=("EDDL_to_Pytorch_LSTM_enc_dec;test_onnx_lstm_enc_dec;import_scripts/recurrent_enc_dec_mnist_pytorch_import.py")
#    eddl2pytorch+=("EDDL_to_Pytorch_GRU_enc_dec;test_onnx_gru_enc_dec;import_scripts/recurrent_enc_dec_mnist_pytorch_import.py")
#    # From EDDL CPU to ONNX RT
#    eddl2pytorch+=("EDDL_to_Pytorch_conv1D_CPU;test_onnx_conv1D_cpu;import_scripts/mnist_pytorch_import.py,--input-1D,--no-channel")
#    eddl2pytorch+=("EDDL_to_Pytorch_conv2D_CPU;test_onnx_conv2D_cpu;import_scripts/mnist_pytorch_import.py,--input-1D,--no-channel")
#    #eddl2pytorch+=("EDDL_to_Pytorch_conv3D_CPU;test_onnx_conv3D_cpu;import_scripts/conv3d_synthetic_pytorch_import.py") TODO: Pytorch synthetic 3D import
#    eddl2pytorch+=("EDDL_to_Pytorch_GRU_imdb_CPU;test_onnx_gru_imdb_cpu;import_scripts/imdb_pytorch_import.py,--unsqueeze-input")
#    eddl2pytorch+=("EDDL_to_Pytorch_LSTM_imdb_CPU;test_onnx_lstm_imdb_cpu;import_scripts/imdb_pytorch_import.py,--unsqueeze-input")
#    eddl2pytorch+=("EDDL_to_Pytorch_LSTM_enc_dec_CPU;test_onnx_lstm_enc_dec_cpu;import_scripts/recurrent_enc_dec_mnist_pytorch_import.py")
#    eddl2pytorch+=("EDDL_to_Pytorch_GRU_enc_dec_CPU;test_onnx_gru_enc_dec_cpu;import_scripts/recurrent_enc_dec_mnist_pytorch_import.py")
#
#    # Run "EDDL export -> Pytorch import" tests and store results
#    print_header "EDDL export -> Pytorch import" >> $tests_results_path
#    for test_data in "${eddl2pytorch[@]}"
#    do
#        IFS=';'; read -r -a test_arr <<< "${test_data}"  # Split string by ";"
#
#        # Get test name to show
#        test_name="${test_arr[0]}"
#
#        # Get paths to onnx file and target metric to get in test
#        model_name="${test_arr[1]}"
#        model_path="${eddl_bin}/model_${model_name}.onnx"
#        model_metric="${eddl_bin}/metric_${model_name}.txt"
#
#        # Get script path and arguments to execute it
#        script_args="${test_arr[2]}"
#        IFS=','; read -r -a script_argv <<< "${script_args}"  # Split string by ","
#        script_call="python ${script_argv[@]} --onnx-file ${model_path} --target-metric ${model_metric}"
#
#        # Execute test
#        echo "Running $test_name"
#        print_header "$script_call" >> ${scripts_output_path}
#        python ${script_argv[0]} ${script_argv[@]:1} --onnx-file ${model_path} --target-metric ${model_metric} &>> ${scripts_output_path}
#        handle_exit_status "$test_name" ${tests_results_path}
#        echo ""
#    done
#
#    popd > /dev/null  # Go back to py_scripts main folder

    #-------------------------------
    # Pytorch export -> EDDL import
    #-------------------------------

    pushd pytorch > /dev/null  # Going to execute Pytorch export scripts to later import with EDDL

    # pytorch2eddl is a list of lists to define the tests for exporting with Pytorch and importing with EDDL
    # Each postion of the inner lists is:
    #   0: Experiment name to show.
    #   1: ONNX model name to import (created by Pytorch).
    #   2: Pytorch export script to create the ONNX model.
    #   3: EDDL script and arguments tu execute for the import test.
    # Note:
    #   - To simulate the list of triplets, the triplets are strings with the elements separated by ";".
    #   - The script names and the arguments to execute them are separated by ",".
    pytorch2eddl=()
    # Pytorch -> EDDL
    pytorch2eddl+=("Pytorch_to_EDDL_conv1D;test_onnx_pytorch_conv1D;export_scripts/conv1D_pytorch_export.py;test_onnx_conv1D,--import")
    pytorch2eddl+=("Pytorch_to_EDDL_conv2D;test_onnx_pytorch_conv2D;export_scripts/conv2D_pytorch_export.py;test_onnx_conv2D,--import")
    pytorch2eddl+=("Pytorch_to_EDDL_conv3D;test_onnx_pytorch_conv3D;export_scripts/conv3D_pytorch_export.py;test_onnx_conv3D,--import")
    pytorch2eddl+=("Pytorch_to_EDDL_convT2D;test_onnx_pytorch_convT2D;export_scripts/convT2D_enc_dec_mnist_pytorch_export.py;test_onnx_convT2D,--import")
    pytorch2eddl+=("Pytorch_to_EDDL_LSTM_IMDB;test_onnx_pytorch_LSTM_imdb;export_scripts/lstm_pytorch_export.py;test_onnx_lstm_imdb,--import")
    pytorch2eddl+=("Pytorch_to_EDDL_GRU_IMDB;test_onnx_pytorch_GRU_imdb;export_scripts/gru_pytorch_export.py;test_onnx_gru_imdb,--import")
    pytorch2eddl+=("Pytorch_to_EDDL_LSTM_MNIST;test_onnx_pytorch_LSTM_mnist;export_scripts/lstm_mnist_pytorch_export.py;test_onnx_lstm_mnist,--import")
    pytorch2eddl+=("Pytorch_to_EDDL_GRU_MNIST;test_onnx_pytorch_GRU_mnist;export_scripts/gru_mnist_pytorch_export.py;test_onnx_gru_mnist,--import")
    pytorch2eddl+=("Pytorch_to_EDDL_RNN_MNIST;test_onnx_pytorch_RNN_mnist;export_scripts/rnn_mnist_pytorch_export.py;test_onnx_rnn_mnist,--import")
    pytorch2eddl+=("Pytorch_to_EDDL_LSTM_enc_dec;test_onnx_pytorch_LSTM_enc_dec;export_scripts/lstm_enc_dec_mnist_pytorch_export.py;test_onnx_lstm_enc_dec,--import")
    pytorch2eddl+=("Pytorch_to_EDDL_GRU_enc_dec;test_onnx_pytorch_GRU_enc_dec;export_scripts/gru_enc_dec_mnist_pytorch_export.py;test_onnx_gru_enc_dec,--import")
    # Pytorch -> EDDL CPU
    #   Note: The export script is set to "none" because we don't need to execute then again
    pytorch2eddl+=("Pytorch_to_EDDL_conv1D_CPU;test_onnx_pytorch_conv1D;none;test_onnx_conv1D,--import,--cpu")
    pytorch2eddl+=("Pytorch_to_EDDL_conv2D_CPU;test_onnx_pytorch_conv2D;none;test_onnx_conv2D,--import,--cpu")
    pytorch2eddl+=("Pytorch_to_EDDL_conv3D_CPU;test_onnx_pytorch_conv3D;none;test_onnx_conv3D,--import,--cpu")
    #pytorch2eddl+=("Pytorch_to_EDDL_convT2D_CPU;test_onnx_pytorch_convT2D;none;test_onnx_convT2D,--import,--cpu") Convt2D not available in CPU
    pytorch2eddl+=("Pytorch_to_EDDL_LSTM_IMDB_CPU;test_onnx_pytorch_LSTM_imdb;none;test_onnx_lstm_imdb,--import,--cpu")
    pytorch2eddl+=("Pytorch_to_EDDL_GRU_IMDB_CPU;test_onnx_pytorch_GRU_imdb;none;test_onnx_gru_imdb,--import,--cpu")
    pytorch2eddl+=("Pytorch_to_EDDL_LSTM_MNIST_CPU;test_onnx_pytorch_LSTM_mnist;none;test_onnx_lstm_mnist,--import,--cpu")
    pytorch2eddl+=("Pytorch_to_EDDL_GRU_MNIST_CPU;test_onnx_pytorch_GRU_mnist;none;test_onnx_gru_mnist,--import,--cpu")
    pytorch2eddl+=("Pytorch_to_EDDL_RNN_MNIST_CPU;test_onnx_pytorch_RNN_mnist;none;test_onnx_rnn_mnist,--import,--cpu")
    pytorch2eddl+=("Pytorch_to_EDDL_LSTM_enc_dec_CPU;test_onnx_pytorch_LSTM_enc_dec;none;test_onnx_lstm_enc_dec,--import,--cpu")
    pytorch2eddl+=("Pytorch_to_EDDL_GRU_enc_dec_CPU;test_onnx_pytorch_GRU_enc_dec;none;test_onnx_gru_enc_dec,--import,--cpu")

    # Run "Pytorch export -> EDDL import" tests and store results
    print_header "Pytorch export -> EDDL import" >> $tests_results_path
    for test_data in "${pytorch2eddl[@]}"
    do
        IFS=';'; read -r -a test_arr <<< "${test_data}"  # Split string by ";"

        # Get test name to show
        test_name="${test_arr[0]}"
        echo "Running $test_name"

        # Get paths of onnx file and target metric for their creation
        model_name="${test_arr[1]}"
        model_path="${eddl_bin}/model_${model_name}.onnx"
        simp_model_path="${eddl_bin}/model_${model_name}_simplified.onnx"  # Model processed with ONNX Simplifier
        model_metric="${eddl_bin}/metric_${model_name}.txt"

        # Get Pytorch export script path and arguments to execute it
        script_args="${test_arr[2]}"
        if [ "$script_args" != "none" ]  # The exported ONNX can be already created
        then
            IFS=','; read -r -a script_argv <<< "${script_args}"  # Split string by ","
            script_call="python ${script_argv[@]} --output-path ${model_path} --output-metric ${model_metric}"

            # Execute Pytorch export
            print_header "$script_call" >> ${scripts_output_path}
            python ${script_argv[0]} ${script_argv[@]:1} --output-path ${model_path} --output-metric ${model_metric} &>> ${scripts_output_path}
            echo "Going to simplify the ONNX model..." &>> ${scripts_output_path}
            python -m onnxsim ${model_path} ${simp_model_path} &>> ${scripts_output_path}  # Simplify model before EDDL import
        else
            echo "  - The export script is set to \"none\". Going to perform the import directly."
        fi

        # Get EDDl import script path and arguments to execute it
        script_args="${test_arr[3]}"
        IFS=','; read -r -a script_argv <<< "${script_args}"  # Split string by ","
        script_call="${script_argv[@]} --onnx-file ${simp_model_path} --target-metric ${model_metric}"

        # Execute EDDL import
        print_header "$script_call" >> ${scripts_output_path}
        pushd ${eddl_bin} > /dev/null  # Go to EDDL binaries folder to execute the import
        ./${script_argv[@]} --onnx-file ${simp_model_path} --target-metric ${model_metric} &>> ${scripts_output_path}
        handle_exit_status "$test_name" ${tests_results_path}
        popd > /dev/null # Go back to pytorch folder
        echo ""
    done

    popd > /dev/null  # Go back to py_scripts main folder

    #-------------------------------
    # Keras export -> EDDL import
    #-------------------------------

    pushd keras > /dev/null  # Going to execute Keras export scripts to later import with EDDL

    # keras2eddl is a list of lists to define the tests for exporting with Keras and importing with EDDL
    # Each postion of the inner lists is:
    #   0: Experiment name to show.
    #   1: ONNX model name to import (created by Keras).
    #   2: Keras export script to create the ONNX model.
    #   3: EDDL script and arguments tu execute for the import test.
    # Note:
    #   - To simulate the list of triplets, the triplets are strings with the elements separated by ";".
    #   - The script names and the arguments to execute them are separated by ",".
    keras2eddl=()
    # Keras -> EDDL
    keras2eddl+=("Keras_to_EDDL_conv1D;test_onnx_keras_conv1D;export_scripts/conv1D_keras_export.py;test_onnx_conv1D,--import")
    keras2eddl+=("Keras_to_EDDL_conv2D;test_onnx_keras_conv2D;export_scripts/conv2D_keras_export.py;test_onnx_conv2D,--import")
    keras2eddl+=("Keras_to_EDDL_conv3D;test_onnx_keras_conv3D;export_scripts/conv3D_keras_export.py;test_onnx_conv3D,--import,--channels-last")
    keras2eddl+=("Keras_to_EDDL_convT2D;test_onnx_keras_convT2D;export_scripts/convT2D_enc_dec_mnist_keras_export.py;test_onnx_convT2D,--import")
    keras2eddl+=("Keras_to_EDDL_LSTM_IMDB;test_onnx_keras_LSTM_imdb;export_scripts/lstm_keras_export.py;test_onnx_lstm_imdb,--import")
    keras2eddl+=("Keras_to_EDDL_GRU_IMDB;test_onnx_keras_GRU_imdb;export_scripts/gru_keras_export.py;test_onnx_gru_imdb,--import")
    keras2eddl+=("Keras_to_EDDL_LSTM_MNIST;test_onnx_keras_LSTM_mnist;export_scripts/lstm_mnist_keras_export.py;test_onnx_lstm_mnist,--import")
    keras2eddl+=("Keras_to_EDDL_GRU_MNIST;test_onnx_keras_GRU_mnist;export_scripts/gru_mnist_keras_export.py;test_onnx_gru_mnist,--import")
    keras2eddl+=("Keras_to_EDDL_RNN_MNIST;test_onnx_keras_RNN_mnist;export_scripts/rnn_mnist_keras_export.py;test_onnx_rnn_mnist,--import")
    keras2eddl+=("Keras_to_EDDL_LSTM_enc_dec;test_onnx_keras_LSTM_enc_dec;export_scripts/lstm_enc_dec_mnist_keras_export.py;test_onnx_lstm_enc_dec,--import")
    keras2eddl+=("Keras_to_EDDL_GRU_enc_dec;test_onnx_keras_GRU_enc_dec;export_scripts/gru_enc_dec_mnist_keras_export.py;test_onnx_gru_enc_dec,--import")
    # Keras -> EDDL CPU
    #   Note: The export script is set to "none" because we don't need to execute then again
    keras2eddl+=("Keras_to_EDDL_conv1D_CPU;test_onnx_keras_conv1D;none;test_onnx_conv1D,--import,--cpu")
    keras2eddl+=("Keras_to_EDDL_conv2D_CPU;test_onnx_keras_conv2D;none;test_onnx_conv2D,--import,--cpu")
    keras2eddl+=("Keras_to_EDDL_conv3D_CPU;test_onnx_keras_conv3D;none;test_onnx_conv3D,--import,--cpu,--channels-last")
    #keras2eddl+=("Keras_to_EDDL_convT2D_CPU;test_onnx_keras_convT2D;none;test_onnx_convT2D,--import,--cpu") Convt2D not available in CPU
    keras2eddl+=("Keras_to_EDDL_LSTM_IMDB_CPU;test_onnx_keras_LSTM_imdb;none;test_onnx_lstm_imdb,--import,--cpu")
    keras2eddl+=("Keras_to_EDDL_GRU_IMDB_CPU;test_onnx_keras_GRU_imdb;none;test_onnx_gru_imdb,--import,--cpu")
    keras2eddl+=("Keras_to_EDDL_LSTM_MNIST_CPU;test_onnx_keras_LSTM_mnist;none;test_onnx_lstm_mnist,--import,--cpu")
    keras2eddl+=("Keras_to_EDDL_GRU_MNIST_CPU;test_onnx_keras_GRU_mnist;none;test_onnx_gru_mnist,--import,--cpu")
    keras2eddl+=("Keras_to_EDDL_RNN_MNIST_CPU;test_onnx_keras_RNN_mnist;none;test_onnx_rnn_mnist,--import,--cpu")
    keras2eddl+=("Keras_to_EDDL_LSTM_enc_dec_CPU;test_onnx_keras_LSTM_enc_dec;none;test_onnx_lstm_enc_dec,--import,--cpu")
    keras2eddl+=("Keras_to_EDDL_GRU_enc_dec_CPU;test_onnx_keras_GRU_enc_dec;none;test_onnx_gru_enc_dec,--import,--cpu")

    # Run "Keras export -> EDDL import" tests and store results
    print_header "Keras export -> EDDL import" >> $tests_results_path
    for test_data in "${keras2eddl[@]}"
    do
        IFS=';'; read -r -a test_arr <<< "${test_data}"  # Split string by ";"

        # Get test name to show
        test_name="${test_arr[0]}"
        echo "Running $test_name"

        # Get paths of onnx file and target metric for their creation
        model_name="${test_arr[1]}"
        model_path="${eddl_bin}/model_${model_name}.onnx"
        model_metric="${eddl_bin}/metric_${model_name}.txt"

        # Get Keras export script path and arguments to execute it
        script_args="${test_arr[2]}"
        if [ "$script_args" != "none" ]  # The exported ONNX can be already created
        then
            IFS=','; read -r -a script_argv <<< "${script_args}"  # Split string by ","
            script_call="python ${script_argv[@]} --output-path ${model_path} --output-metric ${model_metric}"

            # Execute Keras export
            print_header "$script_call" >> ${scripts_output_path}
            python ${script_argv[0]} ${script_argv[@]:1} --output-path ${model_path} --output-metric ${model_metric} &>> ${scripts_output_path}
        else
            echo "  - The export script is set to \"none\". Going to perform the import directly."
        fi

        # Get EDDl import script path and arguments to execute it
        script_args="${test_arr[3]}"
        IFS=','; read -r -a script_argv <<< "${script_args}"  # Split string by ","
        script_call="${script_argv[@]} --onnx-file ${model_path} --target-metric ${model_metric}"

        # Execute EDDL import
        print_header "$script_call" >> ${scripts_output_path}
        pushd ${eddl_bin} > /dev/null  # Go to EDDL binaries folder to execute the import
        ./${script_argv[@]} --onnx-file ${model_path} --target-metric ${model_metric} &>> ${scripts_output_path}
        handle_exit_status "$test_name" ${tests_results_path}
        popd > /dev/null # Go back to keras folder
        echo ""
    done

    popd > /dev/null  # Go back to py_scripts dir
    popd > /dev/null  # Go back to build dir
else
    echo "EDDL only mode activated. Skipping tests with third party libraries"
fi  # $only_eddl -eq 0

echo "#############################################################################"
echo "#"
echo "# To see the tests results look at file $tests_results_filename"
echo "#"
echo "# To see the full experiments outputs look at file $scripts_output_filename"
echo "#"
echo "#############################################################################"

exit 0
