
#include <iostream>
#include "eddl/apis/eddl.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace eddl;


Tensor* preprocess_input_resnet34(Tensor* input, const vector<int> &target_size){
    // Define preprocessing constants
    auto* mean_vec = new Tensor( {0.485, 0.456, 0.406}, {3}, input->device);
    auto* std_vec = new Tensor( {0.229, 0.224, 0.225}, {3}, input->device);

    // ==========================================================================
    // ====== SANITY CHECKS =====================================================
    // ==========================================================================
    // Check dimension. Input must be a 3D or 4D tensor
    if(!(input->ndim == 3 || input->ndim == 4)){
        throw std::runtime_error("A 3D or 4D tensor is expected. " + std::to_string(input->ndim) + "D tensor received.");
    }

    // Convert from 3D to 4D (if needed)
    if(input->ndim == 3){
        input->unsqueeze_(0);
    }
    // ==========================================================================


    // ==========================================================================
    // ====== NORMALIZATION =====================================================
    // ==========================================================================

    // Resize tensor (creates a new instance)
    Tensor* new_input = input->scale(target_size);  // (height, width)

    // Normalization [0..1]
    new_input->mult_(1/255.0f);

    // Standarization: (X-mean)/std
    Tensor* mean = Tensor::broadcast(mean_vec, new_input);
    Tensor* std = Tensor::broadcast(std_vec, new_input);
    new_input->sub_(mean);
    new_input->div_(std);
    // ==========================================================================

    // Free memory
    delete mean_vec;
    delete std_vec;
    delete mean;
    delete std;

    return new_input;
}


int main(int argc, char **argv) {
    // ==========================================================================
    // ====== SET DEFAULT VARIABLES =============================================
    // ==========================================================================

    // Step 0: Download the model, the classes and the image we want to classify
    string image_fname = "../../examples/data/elephant.jpg";
    string class_names_file = "../../examples/data/imagenet_class_names.txt";
    string model_path = "models/resnet34-v1-7.onnx";

    // Step 1: Specify input
    int in_channels = 3;
    int in_height = 224;
    int in_width = 224;
    // ==========================================================================


    // ==========================================================================
    // ====== LOAD ONNX MODEL ===================================================
    // ==========================================================================

    // Import ONNX model
    std::cout << "Importing ONNX..." << std::endl;
    Net *net = import_net_from_onnx_file(model_path, {in_channels, in_height, in_width});
    // ==========================================================================


    // ==========================================================================
    // ====== APPLY POSTPROCESSING ==============================================
    // ==========================================================================

    // Step 4 (optional): Add a softmax layer to get probabilities directly from the model, since it
    // does not include the softmax layer.
    layer input = net->lin[0];   // getLayer(net,"input_layer_name");
    layer output = net->lout[0];   // getLayer(net,"output_layer_name");
    layer new_output = Softmax(output);

    // Create model
    net = Model({input},{new_output});
    // ==========================================================================

    // ==========================================================================
    // ====== COMPILE MODEL =====================================================
    // ==========================================================================

    // Build model
    build(net,
          adam(0.001f), // Optimizer (not used for prediction)
          {"softmax_cross_entropy"}, // Losses (not used for prediction)
          {"categorical_accuracy"}, // Metrics (not used for prediction)
          CS_GPU({1}), // Use one GPU
          false       // Disable model initialization, since we want to use the onnx weights
    );

    // Print and plot our model
    net->summary();
    net->plot();
    // ==========================================================================


    // ==========================================================================
    // ====== INFERENCE =========================================================
    // ==========================================================================
    // Load test image
    Tensor *image = Tensor::load(image_fname);

    // Step 3: Preprocess input. (Look up the preprocessing required at the model's page)
    Tensor* image_preprocessed = preprocess_input_resnet34(image, {in_height, in_width});

    // Predict image. Returns a vector of tensors (here one).
    vector<Tensor*> outputs = net->predict({image_preprocessed});
    // ==========================================================================


    // ==========================================================================
    // ====== PRINT TOP K CLASSES ===============================================
    // ==========================================================================
    // Read imagenet class names from txt file
    std::cout << "Reading imagenet class names..." << std::endl;
    vector<string> class_names = eddl::read_txt_file(class_names_file);

    // Print top K predictions
    int top_k = 5;
    std::cout << "Top " << top_k << " predictions:" << std::endl;
    std::cout << eddl::get_topk_predictions(outputs[0], class_names, top_k)  << std::endl;
    // ==========================================================================

    return 0;
}

