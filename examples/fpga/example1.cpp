/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: December 2021
* Author: GAP Research Group (UPV), contact: jflich@disca.upv.es
* All rights reserved
*/

#include <sys/time.h>

#include "eddl/apis/eddl.h"

using namespace eddl;

int main(int argc, char **argv) { 

  struct timeval time1, time2;
  unsigned long long time_forward;
  layer in, conv;
  model net, net_fpga;
  Tensor *x;

  // Network
  in = Input({64, 256, 256});
  conv = Conv(in, 512, {3, 3}, {1,1}, "same", true);
  
  // Model
  net = Model({in}, {conv});
  build(net);

  // model for fpga
  net_fpga = toFPGA(net, 1, 0);

  // Input data
  x = new Tensor({1, 64, 256, 256});
  x->fill_rand_uniform_(10);

  // forward on FPGA
  reset_profile();
  gettimeofday(&time1, NULL);
  forward(net_fpga, {x}); 
  gettimeofday(&time2, NULL);
  time_forward = ((time2.tv_sec - time1.tv_sec) * 1000000) + (time2.tv_usec - time1.tv_usec);

  // output for fpga
  printf("\n\n------------------------------------------------\n");
  printf("time forward fpga: %12llu usec\n", time_forward);
  summary(net_fpga);
  show_profile();

  // forward on CPU
  reset_profile();
  gettimeofday(&time1, NULL);
  forward(net, {x}); 
  gettimeofday(&time2, NULL);
  time_forward = ((time2.tv_sec - time1.tv_sec) * 1000000) + (time2.tv_usec - time1.tv_usec);

  // output for cpu
  printf("\n\n------------------------------------------------\n");
  printf("time forward cpu:  %12llu usec\n", time_forward);
  summary(net);
  show_profile();
  
  delete net;
  delete x;
  
  return 0;
}
