
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

  
  string image_fname = "../../examples/data/elephant.jpg";
  string class_names_file = "../../examples/data/imagenet_class_names.txt";

  download_hlsinf(1, 0);
  
  if (argc != 2) {
    printf("Usage:\n%s <model>\n", argv[0]);
    printf("  - model 0: VGG16\n");
    printf("  - model 1: VGG16_BN\n");
    printf("  - model 2: VGG19\n");
    printf("  - model 3: VGG19_BN\n");
    printf("  - model 4: RESNET18\n");
    printf("  - model 5: RESNET34\n");
    printf("  - model 6: RESNET50\n");
    printf("  - model 7: RESNET101\n");
    printf("  - model 8: RESNET151\n");
    printf("  - model 9: DENSENET121\n");
    exit(1);
  }
  int i = atoi(argv[1]);
  
  model net;
  switch (i) {
    case 0: net = download_vgg16(false, {3, 224, 224}); break;
    case 1: net = download_vgg16_bn(false, {3, 224, 224}); break;
    case 2: net = download_vgg19(false, {3, 224, 224}); break;
    case 3: net = download_vgg19_bn(false, {3, 224, 224}); break;
    case 4: net = download_resnet18(false, {3, 224, 224}); break;
    case 5: net = download_resnet34(false, {3, 224, 224}); break;
    case 6: net = download_resnet50(false, {3, 224, 224}); break;
    case 7: net = download_resnet101(false, {3, 224, 224}); break;
    case 8: net = download_resnet152(false, {3, 224, 224}); break;
    case 9: net = download_densenet121(false, {3, 224, 224}); break;
  }

  // Add a softmax layer to get probabilities directly from the model, since it does not include the softmax layer.
  layer input = net->lin[0];
  layer output = net->lout[0];
  layer new_output = Softmax(output);

  // Create model
  net = Model({input},{new_output});

  // Build model
  build(net, NULL, CS_CPU({1}), false);

  // Model adaptation to FPGA
  model net_fpga = toFPGA(net);

  // Print model
  net_fpga->summary();

  reset_profile();

  // Load test image
  Tensor *image = Tensor::load(image_fname);

  // Preprocess input. (Look up the preprocessing required at the model's page)
  Tensor* image_preprocessed = preprocess_input_resnet34(image, {224, 224});

  // Predict image. Returns a vector of tensors (here one).
  vector<Tensor*> outputs = net_fpga->predict({image_preprocessed});

  // Read imagenet class names from txt file
  std::cout << "Reading imagenet class names..." << std::endl;
  vector<string> class_names = eddl::read_txt_file(class_names_file);

  // Print top K predictions
  int top_k = 5;
  std::cout << "Top " << top_k << " predictions:" << std::endl;
  std::cout << eddl::get_topk_predictions(outputs[0], class_names, top_k)  << std::endl;

  show_profile();

  return 0;
}




