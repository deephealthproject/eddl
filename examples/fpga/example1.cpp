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

  printf("hell ow world\n\n\n");

  layer in, conv;
  model net, net_fpga;
  Tensor *x;

  download_hlsinf(2, 0);

  // Network
  in = Input({64, 256, 256});
  conv = Conv(in, 512, {3, 3}, {1,1}/*,"same", true*/);
  
  // Model
  net = Model({in}, {conv});
  build(net);

  // model for fpga
  net_fpga = toFPGA(net, 2, 0);

  // Input data
  x = new Tensor({1, 64, 256, 256});
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
  
  delete x;
  delete net;
  delete net_fpga;
  
  return 0;
}
