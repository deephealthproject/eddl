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


int main(int argc, char **argv) {
    // Set default vars
    string image_fname = "../../examples/data/elephant.jpg";  // Some image
    string output = "./";

    // Get model and set its input shape
    // Model from: https://github.com/onnx/models/tree/master/vision/classification/resnet
    string model_path = "/home/salva/Downloads/models/resnet34-v1-7.onnx";
    int in_channels = 3;
    int in_height = 224;
    int in_width = 224;

    std::cout << "Importing ONNX..." << std::endl;
    Net *net = import_net_from_onnx_file(model_path, {in_channels, in_height, in_width}, DEV_CPU);  // Why is the device needed?

    // Optional: This model does not include a softmax layer, so we need to add it *****************
    // Get input/output + Add softmax
    layer in = getLayer(net,"data");  // net->lin[0]
    layer l = getLayer(net,"resnetv16_dense0_fwd");  // net->lout[0]
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

    // Preprocess image as the model wants it ***************************
    // Source: https://github.com/onnx/models/tree/master/vision/classification/resnet

    // Add batch dimension and resize image
    image->unsqueeze_(0);
    Tensor *new_image = image->scale({in_height, in_width});

    // Scale to range [0..1]
    new_image->mult_(1/255.0f);
    // ******************************************************************

    // Predict image
    vector<Tensor*> outputs = net->predict({new_image});

    //Show probabilities
    std::cout << "Class probabilities:" << std::endl;
    outputs[0]->print(2);
    int best_class = outputs[0]->argmax();
    float best_class_prob = outputs[0]->ptr[best_class];
    std::cout << std::setprecision(2) << std::fixed;
    std::cout << "Selected class: " << best_class << " (" << (best_class_prob * 100.0f) << "%)" << std::endl;

	return 0;
}


///////////
