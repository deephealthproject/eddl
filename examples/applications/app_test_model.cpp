
#include <iostream>
#include "eddl/apis/eddl.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace eddl;


Tensor* preprocess_input(Tensor* input, const vector<int> &target_size, bool normalize=true, bool standarize=true){
    // Define preprocessing constants
    auto* mean_vec = new Tensor( {0.485, 0.456, 0.406}, {3, 1}, input->device);
    auto* std_vec = new Tensor( {0.229, 0.224, 0.225}, {3, 1}, input->device);

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
    if(normalize){
        new_input->mult_(1/255.0f);
    }

    // Standarization: (X-mean)/std
    if(standarize){
//    Tensor* mean = Tensor::broadcast(mean_vec, new_input);
//    Tensor* std = Tensor::broadcast(std_vec, new_input);
        // 1) [There is no broadcasting...] Repeat dimensions  => Temp!
        Tensor* mean = Tensor::repeat(mean_vec, target_size[0]*target_size[1], 1);  mean->reshape_(new_input->shape);
        Tensor* std =  Tensor::repeat(std_vec, target_size[0]*target_size[1], 1); std->reshape_(new_input->shape);
        new_input->sub_(mean);
        new_input->div_(std);

        // Free memory
        delete mean;
        delete std;
    }
    // ==========================================================================

    // Free memory
    delete mean_vec;
    delete std_vec;

    return new_input;
}


int main(int argc, char **argv) {
    // ==========================================================================
    // ====== SET DEFAULT VARIABLES =============================================
    // ==========================================================================

    // Step 0: Download the model, the classes and the image we want to classify
    string image_fname = "../../examples/data/elephant.jpg";
    string class_names_file = "../../examples/data/imagenet_class_names.txt";

    // Image Classification
    string model_path = "models/resnet34-v1-7.onnx";  // 3x224x224  // okay
//    string model_path = "models/mobilenetv2-7.onnx"; // 3x224x224 // Signal: SIGSEGV (Segmentation fault)
//    string model_path = "models/vgg16-7.onnx";  // 3xHxW  // okay
//    string model_path = "models/bvlcalexnet-3.onnx";  // 3x224x224  // The onnx node 'LRN' is not supported yet
//    string model_path = "models/bvlcalexnet-12.onnx";  // 3x224x224  // The onnx node 'LRN' is not supported yet
//    string model_path = "models/googlenet-3.onnx";  // 3x224x224  // The onnx node 'LRN' is not supported yet
//    string model_path = "models/densenet-3.onnx";  // 3x224x224  // okay
//    string model_path = "models/inception-v1-3.onnx";  // 3x224x224  // The onnx node 'LRN' is not supported yet
//    string model_path = "models/efficientnet-lite4-11.onnx";  // 224x224x3  // The onnx node 'LRN' is not supported yet

    // Object Detection & Image Segmentation
//    string model_path = "models/tinyyolov2-7.onnx";  // 3x416x416 //Error in Add node Add. The first dimension (batch) of the constant operator must be 1, got 3 (ONNX::ImportNet) ⚠️
//    string model_path = "models/ssd-10_simp.onnx";  // 3x1200x1200 // Signal: SIGSEGV (Segmentation fault)
//    string model_path = "models/FasterRCNN-10_simp.onnx";  // 3xHxW // Signal: SIGSEGV (Segmentation fault)
//    string model_path = "models/yolov4_simp.onnx";  // 416x416x3 // [ONNX::LOG] Detected a padding asymmetry + The onnx node 'Shape' is not supported yet
//    string model_path = "models/MaskRCNN-10_simp.onnx";  // 3xHxW // Signal: SIGSEGV (Segmentation fault)
//    string model_path = "models/retinanet-9.onnx";  // 3xHxW // okay
//    string model_path = "models/yolov3-10_simp.onnx";  // 3x416x416 // The imported model has more than 1 input layer and the shape provided to reshape the model can only be appliedif the model has only one input layer. (ONNX::ImportNet)
//    string model_path = "models/fcn-resnet50-11_simp.onnx";  // 3xHxW // The onnx node 'Shape' is not supported yet
//    string model_path = "models/ResNet101-DUC-7.onnx";  // 3xHxW // okay. Error with plot

    // Image Manipulation
//    string model_path = "models/super-resolution-10.onnx";  // 3x224x224 // Tensors with different size (Tensor::copy)
//    string model_path = "models/mosaic-9.onnx";  // 3xHxW //  Error importing layer . Only "constant" mode is supported (passed "reflect"). (ONNX::ImportNet)

    // Step 1: Specify input
    int in_channels = 3;
    int in_height = 224;
    int in_width = 224;
    vector<int> input_shape = {in_channels, in_height, in_width};
    vector<int> dimensions_order = {0, 1, 2, 3};
    // ==========================================================================


    // ==========================================================================
    // ====== LOAD ONNX MODEL ===================================================
    // ==========================================================================

    // Import ONNX model
    std::cout << "Importing ONNX..." << std::endl;
    Net *net = import_net_from_onnx_file(model_path, input_shape);

    // ==========================================================================
    // Print and plot our model
    net->summary();
    net->plot("mymodel.pdf");

    // ==========================================================================
    // ====== APPLY POSTPROCESSING ==============================================
    // ==========================================================================

    // Step 4 (optional): Add a softmax layer to get probabilities directly from the model, since it
    // does not include the softmax layer.
//    layer input = net->lin[0];   // getLayer(net,"input_layer_name");
//    layer output = net->lout[0];   // getLayer(net,"output_layer_name");
//    layer new_output = Softmax(output);
//
//    // Create model
//    net = Model({input},{new_output});
    // ==========================================================================

    // ==========================================================================
    // ====== COMPILE MODEL =====================================================
    // ==========================================================================

    // Build model
    build(net,
          adam(0.001f), // Optimizer (not used for prediction)
          {"softmax_cross_entropy"},// Losses (not used for prediction)
          {"categorical_accuracy"}, // Metrics (not used for prediction)
          CS_GPU({1}), // Use one GPU
          false       // Disable model initialization, since we want to use the onnx weights
    );
    // ==========================================================================


    // ==========================================================================
    // ====== INFERENCE =========================================================
    // ==========================================================================
    // Load test image
    Tensor *image = Tensor::load(image_fname);

    // Step 3: Preprocess input. (Look up the preprocessing required at the model's page)
    Tensor* image_preprocessed = preprocess_input(image, {in_height, in_width});
    image_preprocessed->permute_(dimensions_order);

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

