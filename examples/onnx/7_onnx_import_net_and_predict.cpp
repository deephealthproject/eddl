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

#include "eddl/apis/eddl.h"
#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed

using namespace eddl;


int main(int argc, char **argv) {
//
//    // Paths
//    string model_path = "/home/salva/Downloads/models/resnet34-v1-7.onnx";
//    string image_fname = "../../examples/data/elephant.jpg";  // Some image
//    string output = "./";
//
//    // Set input shape
//    int in_channels = 3;
//    int in_height = 224;
//    int in_width = 224;
//
//    std::cout << "Importing ONNX..." << std::endl;
//    Net *net = import_net_from_onnx_file(model_path, {in_channels, in_height, in_width}, DEV_CPU);  // Why is the device needed?
//
//    // Print/plot to know layer names
//    net->summary();
////    net->plot();  // Consistency problem
//    plot(net , "default.pdf");  // Not intuitive. Which extension?
//
//    // Get input/output
//    layer in = getLayer(net,"???");
//    layer out = getLayer(net,"???");
//
//    // Create model
//    net = Model({in},{out});
//
//    // Build model
//    build(net,
//          adam(0.001), // Optimizer
//          {"softmax_cross_entropy"}, // Losses
//          {"categorical_accuracy"}, // Metrics
//          CS_GPU({1}), // one GPU
//          false       // Parameter that indicates that the weights of the net must not be initialized to random values.
//    );
//
//
//    // Load test image and resize it
//    Tensor *image = Tensor::load(image_fname);
//    Tensor *new_image = image->scale({in_height, in_width});
//
//    // Predict image
//    net->predict({new_image});
//
//    // Get output (and send it to CPU)
//    Tensor *prediction = out->output->clone(); prediction->toCPU();
//
//    //Show probabilities
//    std::cout << "Class probabilities:" << std::endl;
//    prediction->print(2);
//    int best_class = prediction->argmax();
//    std::cout << "Selected class: " << best_class << std::endl;

	return 0;
}


///////////
