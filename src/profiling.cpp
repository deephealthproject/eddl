/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>

#include "eddl/profiling.h"

#ifdef EDDL_WINDOWS
int gettimeofday(struct timeval* tp, struct timezone* tzp)
{
    namespace sc = std::chrono;
    sc::system_clock::duration d = sc::system_clock::now().time_since_epoch();
    sc::seconds s = sc::duration_cast<sc::seconds>(d);
    tp->tv_sec = s.count();
    tp->tv_usec = sc::duration_cast<sc::microseconds>(d - s).count();

    return 0;
}
#endif


// profiling declarations
PROFILING_ENABLE(maximum);
PROFILING_ENABLE(minimum);
PROFILING_ENABLE(max);
PROFILING_ENABLE(argmax);
PROFILING_ENABLE(argmax_d);
PROFILING_ENABLE(min);
PROFILING_ENABLE(argmin);
PROFILING_ENABLE(sum);
PROFILING_ENABLE(sum_abs);
PROFILING_ENABLE(prod);
PROFILING_ENABLE(mean);
PROFILING_ENABLE(median);
PROFILING_ENABLE(std);
PROFILING_ENABLE(var);
PROFILING_ENABLE(mode);
PROFILING_ENABLE(abs);
PROFILING_ENABLE(acos);
PROFILING_ENABLE(add);
PROFILING_ENABLE(asin);
PROFILING_ENABLE(atan);
PROFILING_ENABLE(cell);
PROFILING_ENABLE(clamp);
PROFILING_ENABLE(clampmax);
PROFILING_ENABLE(clampmin);
PROFILING_ENABLE(cos);
PROFILING_ENABLE(cosh);
PROFILING_ENABLE(div);
PROFILING_ENABLE(exp);
PROFILING_ENABLE(floor);
PROFILING_ENABLE(inv);
PROFILING_ENABLE(log);
PROFILING_ENABLE(log2);
PROFILING_ENABLE(log10);
PROFILING_ENABLE(logn);
PROFILING_ENABLE(mod);
PROFILING_ENABLE(mult);
PROFILING_ENABLE(neg);
PROFILING_ENABLE(normalize);
PROFILING_ENABLE(pow);
PROFILING_ENABLE(powb);
PROFILING_ENABLE(reciprocal);
PROFILING_ENABLE(remainder);
PROFILING_ENABLE(round);
PROFILING_ENABLE(rsqrt);
PROFILING_ENABLE(sigmoid);
PROFILING_ENABLE(sign);
PROFILING_ENABLE(sin);
PROFILING_ENABLE(sinh);
PROFILING_ENABLE(sqr);
PROFILING_ENABLE(sqrt);
PROFILING_ENABLE(sub);
PROFILING_ENABLE(tan);
PROFILING_ENABLE(tanh);
PROFILING_ENABLE(trunc);
PROFILING_ENABLE(inc);
PROFILING_ENABLE(el_div);
PROFILING_ENABLE(mult2D);
PROFILING_ENABLE(el_mult);
PROFILING_ENABLE(sum2D_rowwise);
PROFILING_ENABLE(reduce_sum2D);
PROFILING_ENABLE(sum2D_colwise);
PROFILING_ENABLE(ceil);
// da
PROFILING_ENABLE(shift);
PROFILING_ENABLE(rotate);
PROFILING_ENABLE(scale);
PROFILING_ENABLE(flip);
PROFILING_ENABLE(crop);
PROFILING_ENABLE(crop_scale);
PROFILING_ENABLE(cutout);
PROFILING_ENABLE(shift_random);
PROFILING_ENABLE(rotate_random);
PROFILING_ENABLE(scale_random);
PROFILING_ENABLE(flip_random);
PROFILING_ENABLE(crop_random);
PROFILING_ENABLE(crop_scale_random);
PROFILING_ENABLE(cutout_random);
// reduction
PROFILING_ENABLE(reduce);
PROFILING_ENABLE(reduce_op);
PROFILING_ENABLE(reduction);
PROFILING_ENABLE(reduction_back);
// activations
PROFILING_ENABLE(ELu);
PROFILING_ENABLE(Exp);
PROFILING_ENABLE(ReLu);
PROFILING_ENABLE(Tanh);
PROFILING_ENABLE(D_ELu);
PROFILING_ENABLE(D_Exp);
PROFILING_ENABLE(D_Tanh);
PROFILING_ENABLE(D_ThresholdedReLu);
PROFILING_ENABLE(D_HardSigmoid);
PROFILING_ENABLE(D_LeakyRelu);
PROFILING_ENABLE(D_Linear);
PROFILING_ENABLE(D_ReLu);
PROFILING_ENABLE(D_LeakyReLu);
PROFILING_ENABLE(D_Sigmoid);
PROFILING_ENABLE(D_Softmax);
PROFILING_ENABLE(D_softplus);
PROFILING_ENABLE(HardSigmoid);
PROFILING_ENABLE(D_softsign);
PROFILING_ENABLE(LeakyReLu);
PROFILING_ENABLE(Linear);
PROFILING_ENABLE(Sigmoid);
PROFILING_ENABLE(Softmax);
PROFILING_ENABLE(Softplus);
PROFILING_ENABLE(Softsign);
PROFILING_ENABLE(ThresholdedReLu);
// conv
PROFILING_ENABLE(Conv2D);
PROFILING_ENABLE(Conv2DReLU);
PROFILING_ENABLE(Conv2D_grad);
PROFILING_ENABLE(Conv2D_back);
// losses
PROFILING_ENABLE(cent);
// generator
PROFILING_ENABLE(fill_rand_uniform);
PROFILING_ENABLE(fill_rand_signed_uniform);
PROFILING_ENABLE(fill_rand_normal);
PROFILING_ENABLE(fill_rand_binary);
// comparison
PROFILING_ENABLE(all);
PROFILING_ENABLE(any);
PROFILING_ENABLE(isfinite);
PROFILING_ENABLE(isinf);
PROFILING_ENABLE(isnan);
PROFILING_ENABLE(isneginf);
PROFILING_ENABLE(isposinf);
PROFILING_ENABLE(logical_and);
PROFILING_ENABLE(logical_or);
PROFILING_ENABLE(logical_not);
PROFILING_ENABLE(logical_xor);
PROFILING_ENABLE(allclose);
PROFILING_ENABLE(isclose);
PROFILING_ENABLE(greater);
PROFILING_ENABLE(greater_equal);
PROFILING_ENABLE(less);
PROFILING_ENABLE(less_equal);
PROFILING_ENABLE(equal);
PROFILING_ENABLE(not_equal);
PROFILING_ENABLE(equivalent);
// bn
PROFILING_ENABLE(permute_channels_last);
PROFILING_ENABLE(permute_channels_first);
PROFILING_ENABLE(permute_batch_last);
PROFILING_ENABLE(permute_batch_first);
// core_nn
PROFILING_ENABLE(repeat_nn);
PROFILING_ENABLE(d_repeat_nn);
PROFILING_ENABLE(select);
PROFILING_ENABLE(select_back);
PROFILING_ENABLE(set_select);
PROFILING_ENABLE(set_select_back);
PROFILING_ENABLE(transform);
// metrics
PROFILING_ENABLE(accuracy);
PROFILING_ENABLE(bin_accuracy);
// pool
PROFILING_ENABLE(MPool2D);
PROFILING_ENABLE(MPool2D_back);
PROFILING_ENABLE(AvgPool2D);
PROFILING_ENABLE(AvgPool2D_back);
// fpga-specific
PROFILING_ENABLE(fpga_hlsinf);
PROFILING_ENABLE(Precision_Conversion);
PROFILING_ENABLE(FPGA_READ);
PROFILING_ENABLE(FPGA_WRITE);

// training-inference-specific
PROFILING_ENABLE(forward);

void __show_profile() {

  // training-inference-specific
  printf("==============================================================================================================\n");
  printf("| Profiling (model)                                                                                          |\n");
  printf("-------------------------------------------------------------------------------------------------------------|\n");
  
  PROFILING_PRINTF(forward);

  printf("==============================================================================================================\n");
  printf("| Profiling (functions)                                                                                      |\n");
  printf("-------------------------------------------------------------------------------------------------------------|\n");

  // profiling declarations
  PROFILING_PRINTF(maximum);
  PROFILING_PRINTF(minimum);
  PROFILING_PRINTF(max);
  PROFILING_PRINTF(argmax);
  PROFILING_PRINTF(argmax_d);
  PROFILING_PRINTF(min);
  PROFILING_PRINTF(argmin);
  PROFILING_PRINTF(sum);
  PROFILING_PRINTF(sum_abs);
  PROFILING_PRINTF(prod);
  PROFILING_PRINTF(mean);
  PROFILING_PRINTF(median);
  PROFILING_PRINTF(std);
  PROFILING_PRINTF(var);
  PROFILING_PRINTF(mode);
  PROFILING_PRINTF(abs);
  PROFILING_PRINTF(acos);
  PROFILING_PRINTF(add);
  PROFILING_PRINTF(asin);
  PROFILING_PRINTF(atan);
  PROFILING_PRINTF(cell);
  PROFILING_PRINTF(clamp);
  PROFILING_PRINTF(clampmax);
  PROFILING_PRINTF(clampmin);
  PROFILING_PRINTF(cos);
  PROFILING_PRINTF(cosh);
  PROFILING_PRINTF(div);
  PROFILING_PRINTF(exp);
  PROFILING_PRINTF(floor);
  PROFILING_PRINTF(inv);
  PROFILING_PRINTF(log);
  PROFILING_PRINTF(log2);
  PROFILING_PRINTF(log10);
  PROFILING_PRINTF(logn);
  PROFILING_PRINTF(mod);
  PROFILING_PRINTF(mult);
  PROFILING_PRINTF(neg);
  PROFILING_PRINTF(normalize);
  PROFILING_PRINTF(pow);
  PROFILING_PRINTF(powb);
  PROFILING_PRINTF(reciprocal);
  PROFILING_PRINTF(remainder);
  PROFILING_PRINTF(round);
  PROFILING_PRINTF(rsqrt);
  PROFILING_PRINTF(sigmoid);
  PROFILING_PRINTF(sign);
  PROFILING_PRINTF(sin);
  PROFILING_PRINTF(sinh);
  PROFILING_PRINTF(sqr);
  PROFILING_PRINTF(sqrt);
  PROFILING_PRINTF(sub);
  PROFILING_PRINTF(tan);
  PROFILING_PRINTF(tanh);
  PROFILING_PRINTF(trunc);
  PROFILING_PRINTF(inc);
  PROFILING_PRINTF(el_div);
  PROFILING_PRINTF(mult2D);
  PROFILING_PRINTF(el_mult);
  PROFILING_PRINTF(sum2D_rowwise);
  PROFILING_PRINTF(reduce_sum2D);
  PROFILING_PRINTF(sum2D_colwise);
  PROFILING_PRINTF(ceil);
  // da
  PROFILING_PRINTF(shift);
  PROFILING_PRINTF(rotate);
  PROFILING_PRINTF(scale);
  PROFILING_PRINTF(flip);
  PROFILING_PRINTF(crop);
  PROFILING_PRINTF(crop_scale);
  PROFILING_PRINTF(cutout);
  PROFILING_PRINTF(shift_random);
  PROFILING_PRINTF(rotate_random);
  PROFILING_PRINTF(scale_random);
  PROFILING_PRINTF(flip_random);
  PROFILING_PRINTF(crop_random);
  PROFILING_PRINTF(crop_scale_random);
  PROFILING_PRINTF(cutout_random);
  //reduction
  PROFILING_PRINTF(reduce);
  PROFILING_PRINTF(reduce_op);
  PROFILING_PRINTF(reduction);
  PROFILING_PRINTF(reduction_back);
  // activations
  PROFILING_PRINTF(ELu);
  PROFILING_PRINTF(Exp);
  PROFILING_PRINTF(ReLu);
  PROFILING_PRINTF(Tanh);
  PROFILING_PRINTF(D_ELu);
  PROFILING_PRINTF(D_Exp);
  PROFILING_PRINTF(D_Tanh);
  PROFILING_PRINTF(D_ThresholdedReLu);
  PROFILING_PRINTF(D_HardSigmoid);
  PROFILING_PRINTF(D_LeakyRelu);
  PROFILING_PRINTF(D_Linear);
  PROFILING_PRINTF(D_ReLu);
  PROFILING_PRINTF(D_LeakyReLu);
  PROFILING_PRINTF(D_Sigmoid);
  PROFILING_PRINTF(D_Softmax);
  PROFILING_PRINTF(D_softplus);
  PROFILING_PRINTF(HardSigmoid);
  PROFILING_PRINTF(D_softsign);
  PROFILING_PRINTF(LeakyReLu);
  PROFILING_PRINTF(Linear);
  PROFILING_PRINTF(Sigmoid);
  PROFILING_PRINTF(Softmax);
  PROFILING_PRINTF(Softplus);
  PROFILING_PRINTF(Softsign);
  PROFILING_PRINTF(ThresholdedReLu);
  // conv
  PROFILING_PRINTF(Conv2D);
  PROFILING_PRINTF(Conv2DReLU);
  PROFILING_PRINTF(Conv2D_grad);
  PROFILING_PRINTF(Conv2D_back);
  // losses
  PROFILING_PRINTF(cent);
  // generator
  PROFILING_PRINTF(fill_rand_uniform);
  PROFILING_PRINTF(fill_rand_signed_uniform);
  PROFILING_PRINTF(fill_rand_normal);
  PROFILING_PRINTF(fill_rand_binary);
  // comparison
  PROFILING_PRINTF(all);
  PROFILING_PRINTF(any);
  PROFILING_PRINTF(isfinite);
  PROFILING_PRINTF(isinf);
  PROFILING_PRINTF(isnan);
  PROFILING_PRINTF(isneginf);
  PROFILING_PRINTF(isposinf);
  PROFILING_PRINTF(logical_and);
  PROFILING_PRINTF(logical_or);
  PROFILING_PRINTF(logical_not);
  PROFILING_PRINTF(logical_xor);
  PROFILING_PRINTF(allclose);
  PROFILING_PRINTF(isclose);
  PROFILING_PRINTF(greater);
  PROFILING_PRINTF(greater_equal);
  PROFILING_PRINTF(less);
  PROFILING_PRINTF(less_equal);
  PROFILING_PRINTF(equal);
  PROFILING_PRINTF(not_equal);
  PROFILING_PRINTF(equivalent);
  // bn
  PROFILING_PRINTF(permute_channels_last);
  PROFILING_PRINTF(permute_channels_first);
  PROFILING_PRINTF(permute_batch_last);
  PROFILING_PRINTF(permute_batch_first);
  // core_nn
  PROFILING_PRINTF(repeat_nn);
  PROFILING_PRINTF(d_repeat_nn);
  PROFILING_PRINTF(select);
  PROFILING_PRINTF(select_back);
  PROFILING_PRINTF(set_select);
  PROFILING_PRINTF(set_select_back);
  PROFILING_PRINTF(transform);
  // metrics
  PROFILING_PRINTF(accuracy);
  PROFILING_PRINTF(bin_accuracy);
  // pool
  PROFILING_PRINTF(MPool2D);
  PROFILING_PRINTF(MPool2D_back);
  PROFILING_PRINTF(AvgPool2D);

  // fpga-specific
  PROFILING_PRINTF(fpga_hlsinf);
  PROFILING_PRINTF(Precision_Conversion);
  PROFILING_PRINTF(FPGA_READ);
  PROFILING_PRINTF(FPGA_WRITE);

  printf("==============================================================================================================\n");

}

void __reset_profile() {

  // training-inference-specific
  PROFILING_RESET(forward);

  // profiling declarations
  PROFILING_RESET(maximum);
  PROFILING_RESET(minimum);
  PROFILING_RESET(max);
  PROFILING_RESET(argmax);
  PROFILING_RESET(argmax_d);
  PROFILING_RESET(min);
  PROFILING_RESET(argmin);
  PROFILING_RESET(sum);
  PROFILING_RESET(sum_abs);
  PROFILING_RESET(prod);
  PROFILING_RESET(mean);
  PROFILING_RESET(median);
  PROFILING_RESET(std);
  PROFILING_RESET(var);
  PROFILING_RESET(mode);
  PROFILING_RESET(abs);
  PROFILING_RESET(acos);
  PROFILING_RESET(add);
  PROFILING_RESET(asin);
  PROFILING_RESET(atan);
  PROFILING_RESET(cell);
  PROFILING_RESET(clamp);
  PROFILING_RESET(clampmax);
  PROFILING_RESET(clampmin);
  PROFILING_RESET(cos);
  PROFILING_RESET(cosh);
  PROFILING_RESET(div);
  PROFILING_RESET(exp);
  PROFILING_RESET(floor);
  PROFILING_RESET(inv);
  PROFILING_RESET(log);
  PROFILING_RESET(log2);
  PROFILING_RESET(log10);
  PROFILING_RESET(logn);
  PROFILING_RESET(mod);
  PROFILING_RESET(mult);
  PROFILING_RESET(neg);
  PROFILING_RESET(normalize);
  PROFILING_RESET(pow);
  PROFILING_RESET(powb);
  PROFILING_RESET(reciprocal);
  PROFILING_RESET(remainder);
  PROFILING_RESET(round);
  PROFILING_RESET(rsqrt);
  PROFILING_RESET(sigmoid);
  PROFILING_RESET(sign);
  PROFILING_RESET(sin);
  PROFILING_RESET(sinh);
  PROFILING_RESET(sqr);
  PROFILING_RESET(sqrt);
  PROFILING_RESET(sub);
  PROFILING_RESET(tan);
  PROFILING_RESET(tanh);
  PROFILING_RESET(trunc);
  PROFILING_RESET(inc);
  PROFILING_RESET(el_div);
  PROFILING_RESET(mult2D);
  PROFILING_RESET(el_mult);
  PROFILING_RESET(sum2D_rowwise);
  PROFILING_RESET(reduce_sum2D);
  PROFILING_RESET(sum2D_colwise);
  PROFILING_RESET(ceil);
  // da
  PROFILING_RESET(shift);
  PROFILING_RESET(rotate);
  PROFILING_RESET(scale);
  PROFILING_RESET(flip);
  PROFILING_RESET(crop);
  PROFILING_RESET(crop_scale);
  PROFILING_RESET(cutout);
  PROFILING_RESET(shift_random);
  PROFILING_RESET(rotate_random);
  PROFILING_RESET(scale_random);
  PROFILING_RESET(flip_random);
  PROFILING_RESET(crop_random);
  PROFILING_RESET(crop_scale_random);
  PROFILING_RESET(cutout_random);
  //reduction
  PROFILING_RESET(reduce);
  PROFILING_RESET(reduce_op);
  PROFILING_RESET(reduction);
  PROFILING_RESET(reduction_back);
  // activations
  PROFILING_RESET(ELu);
  PROFILING_RESET(Exp);
  PROFILING_RESET(ReLu);
  PROFILING_RESET(Tanh);
  PROFILING_RESET(D_ELu);
  PROFILING_RESET(D_Exp);
  PROFILING_RESET(D_Tanh);
  PROFILING_RESET(D_ThresholdedReLu);
  PROFILING_RESET(D_HardSigmoid);
  PROFILING_RESET(D_LeakyRelu);
  PROFILING_RESET(D_Linear);
  PROFILING_RESET(D_ReLu);
  PROFILING_RESET(D_LeakyReLu);
  PROFILING_RESET(D_Sigmoid);
  PROFILING_RESET(D_Softmax);
  PROFILING_RESET(D_softplus);
  PROFILING_RESET(HardSigmoid);
  PROFILING_RESET(D_softsign);
  PROFILING_RESET(LeakyReLu);
  PROFILING_RESET(Linear);
  PROFILING_RESET(Sigmoid);
  PROFILING_RESET(Softmax);
  PROFILING_RESET(Softplus);
  PROFILING_RESET(Softsign);
  PROFILING_RESET(ThresholdedReLu);
  // conv
  PROFILING_RESET(Conv2D);
  PROFILING_RESET(Conv2DReLU);
  PROFILING_RESET(Conv2D_grad);
  PROFILING_RESET(Conv2D_back);
  // losses
  PROFILING_RESET(cent);
  // generator
  PROFILING_RESET(fill_rand_uniform);
  PROFILING_RESET(fill_rand_signed_uniform);
  PROFILING_RESET(fill_rand_normal);
  PROFILING_RESET(fill_rand_binary);
  // comparison
  PROFILING_RESET(all);
  PROFILING_RESET(any);
  PROFILING_RESET(isfinite);
  PROFILING_RESET(isinf);
  PROFILING_RESET(isnan);
  PROFILING_RESET(isneginf);
  PROFILING_RESET(isposinf);
  PROFILING_RESET(logical_and);
  PROFILING_RESET(logical_or);
  PROFILING_RESET(logical_not);
  PROFILING_RESET(logical_xor);
  PROFILING_RESET(allclose);
  PROFILING_RESET(isclose);
  PROFILING_RESET(greater);
  PROFILING_RESET(greater_equal);
  PROFILING_RESET(less);
  PROFILING_RESET(less_equal);
  PROFILING_RESET(equal);
  PROFILING_RESET(not_equal);
  PROFILING_RESET(equivalent);
  // bn
  PROFILING_RESET(permute_channels_last);
  PROFILING_RESET(permute_channels_first);
  PROFILING_RESET(permute_batch_last);
  PROFILING_RESET(permute_batch_first);
  // core_nn
  PROFILING_RESET(repeat_nn);
  PROFILING_RESET(d_repeat_nn);
  PROFILING_RESET(select);
  PROFILING_RESET(select_back);
  PROFILING_RESET(set_select);
  PROFILING_RESET(set_select_back);
  PROFILING_RESET(transform);
  // metrics
  PROFILING_RESET(accuracy);
  PROFILING_RESET(bin_accuracy);
  // pool
  PROFILING_RESET(MPool2D);
  PROFILING_RESET(MPool2D_back);
  PROFILING_RESET(AvgPool2D);

  // fpga-specific
  PROFILING_RESET(fpga_hlsinf);
  PROFILING_RESET(Precision_Conversion);
  PROFILING_RESET(FPGA_READ);
  PROFILING_RESET(FPGA_WRITE);
}

