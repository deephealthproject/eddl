/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: December 2021
* Author: GAP Research Group (UPV), contact: jflich@disca.upv.es
* All rights reserved
*/

#include "eddl/apis/eddl.h"

using namespace eddl;

int main(int argc, char **argv) { 

  download_hlsinf(1, 0);

  // Network
  layer in = Input({64, 256, 256});
  layer conv = Conv(in, 512, {3, 3}, {1,1}, "same", true);
  
  // Model
  model net = Model({in}, {conv});
  build(net);

  // model for fpga
  model net_fpga = toFPGA(net, 1, 0);

  // Input data
  Tensor *x = new Tensor({1, 64, 256, 256});
  x->fill_rand_uniform_(10);

  // forward on FPGA
  reset_profile();
  forward(net_fpga, {x}); 

  // output for fpga
  summary(net_fpga);
  show_profile();

  // forward on CPU
  reset_profile();
  forward(net, {x}); 

  // output for cpu
  summary(net);
  show_profile();

  Tensor *output = getOutput(net->lout[0]);
  Tensor *output_fpga = getOutput(net_fpga->lout[0]);
  if (output->allclose(output_fpga, 1e-03, 1e-03)) printf("Outputs all_close\n"); else printf("Outputs differ too much\n");
  
  return 0;
}
