/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "eddl/apis/eddl.h"
#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed

using namespace eddl;

Tensor* preprocess_input_resnet34(Tensor* input, const vector<int> &target_size){
    // Preprocess image (depends on the model) ***************************
    // Values from: https://github.com/onnx/models/tree/master/vision/classification/resnet

    auto* mean_vec = new Tensor( {0.485, 0.456, 0.406}, {3, 1}, input->device);
    auto* std_vec = new Tensor( {0.229, 0.224, 0.225}, {3, 1}, input->device);

    // Check dimension. Input must be a 3D or 4D tensor
    if(!(input->ndim == 3 || input->ndim == 4)){
        throw std::runtime_error("A 3D or 4D tensor is expected. " + std::to_string(input->ndim) + "D tensor received.");
    }

    // Convert from 3D to 4D (if needed)
    if(input->ndim == 3){
        input->unsqueeze_(0);
    }

    // Resize tensor (creates a new instance)
    Tensor* new_input = input->scale(target_size);  // (height, width)

    // Scale to range [0..1]
    new_input->mult_(1/255.0f);

    // Normalize: (X-mean)/std
    // 1) [There is no broadcasting...] Repeat dimensions
//    Tensor* mean = Tensor::broadcast(mean_vec, new_input);
//    Tensor* std = Tensor::broadcast(std_vec, new_input);
    Tensor* mean = Tensor::repeat(mean_vec, target_size[0]*target_size[1], 1);  mean->reshape_(new_input->shape);
    Tensor* std =  Tensor::repeat(std_vec, target_size[0]*target_size[1], 1); std->reshape_(new_input->shape);
    new_input->sub_(mean);
    new_input->div_(std);
    // ******************************************************************

    delete mean_vec;
    delete std_vec;
    delete mean;
    delete std;

    return new_input;
}

int main(int argc, char **argv) {
    // [Manual]
    // Model from: https://github.com/onnx/models/tree/master/vision/classification/resnet
    // ResNet34 onnx: https://github.com/onnx/models/blob/master/vision/classification/resnet/model/resnet34-v1-7.onnx

    // Set default vars
    string image_fname = "../../examples/data/elephant.jpg";
    string class_names_file = "../../examples/data/imagenet_class_names.txt";
    string model_path = "/home/salva/Downloads/models/resnet34-v1-7.onnx";

    // Specific
    int in_channels = 3;
    int in_height = 224;
    int in_width = 224;

    // Import ONNX model
    std::cout << "Importing ONNX..." << std::endl;
    Net *net = import_net_from_onnx_file(model_path, {in_channels, in_height, in_width}, DEV_CPU);  // Why is the device needed?

    // **********************************************************************************************
    // Optional: This model does not include a softmax layer, so we need to add it
    // Get input/output + Add softmax
    layer in = net->lin[0];   // getLayer(net,"input_layer_name");
    layer l = net->lout[0];   // getLayer(net,"output_layer_name");
    layer out = Softmax(l);

    // Create model
    net = Model({in},{out});
    // **********************************************************************************************

    // Print/plot to know layer names
    net->summary();
    net->plot("default.pdf", "LR");   // Not intuitive. Which extension? What is "LR"?

    // Build model
    build(net,
          adam(0.001), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}), // one GPU
          false       // Parameter that indicates that the weights of the net must not be initialized to random values.
    );

    // Load test image
    Tensor *image = Tensor::load(image_fname);

    // Preprocess input. (Depends on the model)
    Tensor* image_preprocessed = preprocess_input_resnet34(image, {in_height, in_width});

    // Predict image
    vector<Tensor*> outputs = net->predict({image_preprocessed});

    // Read class names from txt file
    std::cout << "Reading class names..." << std::endl;
    vector<string> class_names = eddl::read_txt_file(class_names_file);

    // Show top K predictions
    int top_k = 5;
    std::cout << "Top " << top_k << " predictions:" << std::endl;
    std::cout << eddl::get_topk_predictions(outputs[0], class_names, top_k)  << std::endl;

	return 0;
}


///////////
