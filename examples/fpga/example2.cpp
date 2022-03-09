#include "eddl/apis/eddl.h"

using namespace eddl;

int main(int argc, char **argv)
{

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

  build(net, nullptr, CS_CPU({1}), false);
  model net_fpga = toFPGA(net, 1, 0);

  summary(net);
  summary(net_fpga);

  Tensor *x = new Tensor({1, 3, 224, 224});
  x->fill_rand_uniform_(10);
  
  reset_profile();
  predict(net_fpga, { x });

  printf("\n\nProfiling (FPGA model run):\n");
  show_profile();

  reset_profile();
  predict(net, { x });

  printf("\n\nProfiling (CPU model run):\n");
  show_profile();

  Tensor *output = getOutput(net->lout[0]);
  Tensor *output_fpga = getOutput(net_fpga->lout[0]);
  if (output->allclose(output_fpga, 1e-03, 1e-03)) printf("Outputs all_close\n"); else printf("Outputs differ too much\n");

  return EXIT_SUCCESS;
}
