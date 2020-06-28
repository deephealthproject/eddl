/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>
#include <limits>       // std::numeric_limits

#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"

// -----------------------------------------------------------------
// mpool2D
//
void fpga_cpuemu_mpool2D(PoolDescriptor *D) {
  fpga_copy_from_fpga(D->I, D->I->ptr);
  cpu_mpool2D(D);
  fpga_copy_to_fpga(D->O->ptr, D->O);
  fpga_copy_memory_to_fpga(D->indX->ptr, D->indX->fpga_ptr, D->indX->size);
  fpga_copy_memory_to_fpga(D->indY->ptr, D->indY->fpga_ptr, D->indY->size);
}

void fpga_mpool2D(PoolDescriptor *D){
  _profile_fpga(_FPGA_MPOOL2D, 0);
  _profile_fpga_tensor(D->I);
#ifndef K_ENABLED_MPOOL2D
  fpga_cpuemu_mpool2D(D);
#else
  cl_int err;
  cl::Event event;

  // parameters
  int batch_size          = D->I->shape[0];     // batch size
  cl::Buffer I            = *D->I->fpga_ptr;    // input 
  int Irows               = D->ir;              // input rows
  int Icols               = D->ic;              // input cols
  int Ichannels           = D->iz;              // input channels
  cl::Buffer O            = *D->O->fpga_ptr;    // output
  cl::Buffer indx         = *D->indX->fpga_ptr; // indices (X component)
  cl::Buffer indy         = *D->indY->fpga_ptr; // indices (Y component)
  int padding_rows_top    = D->padrt;           // padding rows (top)
  int padding_rows_bottom = D->padrb;           // padding rows (bottom)
  int padding_cols_left   = D->padcl;           // padding cols (left)
  int padding_cols_right  = D->padcr;           // padding cols (right)
  int kernel_rows         = D->kr;              // kernel rows
  int kernel_cols         = D->kc;              // kernel cols
  int stride_rows         = D->sr;              // stride rows
  int stride_cols         = D->sc;              // stride cols
  int Dsize               = D->size;            // size (descriptor's field)

  OCL_CHECK(err, err = kernel_mpool2D.setArg(0, batch_size));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(1, I));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(2, Irows));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(3, Icols));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(4, Ichannels));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(5, O));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(6, indx));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(7, indy));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(8, padding_rows_top));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(9, padding_rows_bottom));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(10, padding_cols_left));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(11, padding_cols_right));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(12, kernel_rows));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(13, kernel_cols));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(14, stride_rows));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(15, stride_cols));
  OCL_CHECK(err, err = kernel_mpool2D.setArg(16, Dsize));

  OCL_CHECK(err, err = q.enqueueTask(kernel_mpool2D, NULL, &event));
  q.finish();
#endif
  _profile_fpga_tensor(D->O);
  _profile_fpga(_FPGA_MPOOL2D, 1);
}

// -----------------------------------------------------------------
// mpool2D_back
//
void fpga_cpuemu_mpool2D_back(PoolDescriptor *D) {
  fpga_copy_from_fpga(D->D, D->D->ptr);
  fpga_copy_memory_from_fpga(D->indX->fpga_ptr, D->indX->ptr, D->indX->size);
  fpga_copy_memory_from_fpga(D->indY->fpga_ptr, D->indY->ptr, D->indY->size);
  cpu_mpool2D_back(D);
  fpga_copy_to_fpga(D->ID->ptr, D->ID);
}

void fpga_mpool2D_back(PoolDescriptor *D){
  _profile_fpga(_FPGA_MPOOL2D_BACK, 0);
#ifndef _K_ENABLED_MPOOL2D_BACK
  fpga_cpuemu_mpool2D_back(D);
#else
  cl_int err;
  cl::Event event;

  // parameters
  int batch_size          = D->I->shape[0];     // batch size
  cl::Buffer ID           = *D->ID->fpga_ptr;   // input delta
  int IDrows              = D->ir;              // input rows
  int IDcols              = D->ic;              // input cols
  int IDchannels          = D->iz;              // input channels
  cl::Buffer D            = *D->O->fpga_ptr;    // D
  cl::Buffer indx         = *D->indX->fpga_ptr; // indices (X component)
  cl::Buffer indy         = *D->indY->fpga_ptr; // indices (Y component)
  int padding_rows_top    = D->padrt;           // padding rows (top)
  int padding_rows_bottom = D->padrb;           // padding rows (bottom)
  int padding_cols_left   = D->padcl;           // padding cols (left)
  int padding_cols_right  = D->padcr;           // padding cols (right)
  int kernel_rows         = D->kr;              // kernel rows
  int kernel_cols         = D->kc;              // kernel cols
  int stride_rows         = D->sr;              // stride rows
  int stride_cols         = D->sc;              // stride cols
  int Dsize               = D->size;            // size (descriptor's field)
  
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(0, batch_size));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(1, ID));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(2, IDrows));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(3, IDcols));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(4, IDchannels));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(5, D)); 
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(6, indx));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(7, indy));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(8, padding_rows_top));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(9, padding_rows_bottom));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(10, padding_cols_left));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(11, padding_cols_right));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(12, kernel_rows));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(13, kernel_cols));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(14, stride_rows));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(15, stride_cols));
  OCL_CHECK(err, err = kernel_mpool2D_back.setArg(16, Dsize));

  OCL_CHECK(err, err = q.enqueueTask(kernel_mpool2D_back, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_MPOOL2D_BACK, 1);
}

// -----------------------------------------------------------------
// avgpool2D
//
void fpga_cpuemu_avgpool2D(PoolDescriptor *D) {
  fpga_copy_from_fpga(D->I, D->I->ptr);
  cpu_avgpool2D(D);
  fpga_copy_to_fpga(D->O->ptr, D->O);
}

void fpga_avgpool2D(PoolDescriptor *D){
  _profile_fpga(_FPGA_AVGPOOL2D, 0);
  _profile_fpga_tensor(D->I);
#ifndef K_ENABLED_AVGPOOL2D
  fpga_cpuemu_avgpool2D(D);
#else
  cl_int err;
  cl::Event event;

  // parameters
  int batch_size          = D->I->shape[0];     // batch size
  cl::Buffer I            = *D->I->fpga_ptr;    // input
  int Irows               = D->ir;              // input rows
  int Icols               = D->ic;              // input cols
  int Ichannels           = D->iz;              // input channels
  cl::Buffer O            = *D->O->fpga_ptr;    // output
  int padding_rows_top    = D->padrt;           // padding rows (top)
  int padding_rows_bottom = D->padrb;           // padding rows (bottom)
  int padding_cols_left   = D->padcl;           // padding cols (left)
  int padding_cols_right  = D->padcr;           // padding cols (right)
  int kernel_rows         = D->kr;              // kernel rows
  int kernel_cols         = D->kc;              // kernel cols
  int stride_rows         = D->sr;              // stride rows
  int stride_cols         = D->sc;              // stride cols
  int Dsize               = D->size;            // size (descriptor's field)

  OCL_CHECK(err, err = kernel_avgmpool2D.setArg(0, batch_size));
  OCL_CHECK(err, err = kernel_avgpool2D.setArg(1, I));
  OCL_CHECK(err, err = kernel_avgpool2D.setArg(2, Irows));
  OCL_CHECK(err, err = kernel_avgpool2D.setArg(3, Icols));
  OCL_CHECK(err, err = kernel_avgpool2D.setArg(4, Ichannels));
  OCL_CHECK(err, err = kernel_avgpool2D.setArg(5, O));
  OCL_CHECK(err, err = kernel_avgmpool2D.setArg(6, padding_rows_top));
  OCL_CHECK(err, err = kernel_avgpool2D.setArg(7, padding_rows_bottom));
  OCL_CHECK(err, err = kernel_avgpool2D.setArg(8, padding_cols_left));
  OCL_CHECK(err, err = kernel_avgpool2D.setArg(9, padding_cols_right));
  OCL_CHECK(err, err = kernel_avgpool2D.setArg(10, kernel_rows));
  OCL_CHECK(err, err = kernel_avgpool2D.setArg(11, kernel_cols));
  OCL_CHECK(err, err = kernel_avgpool2D.setArg(12, stride_rows));
  OCL_CHECK(err, err = kernel_avgpool2D.setArg(13, stride_cols));
  OCL_CHECK(err, err = kernel_avgpool2D.setArg(14, Dsize));

  OCL_CHECK(err, err = q.enqueueTask(kernel_avgpool2D, NULL, &event));
  q.finish();
#endif
  _profile_fpga_tensor(D->O);
  _profile_fpga(_FPGA_AVGPOOL2D, 1);
}

// -----------------------------------------------------------------
// avgpool2D_back
//
void fpga_cpuemu_avgpool2D_back(PoolDescriptor *D) {
  fpga_copy_from_fpga(D->D, D->D->ptr);
  cpu_avgpool2D_back(D);
  fpga_copy_to_fpga(D->ID->ptr, D->ID);
}

void fpga_avgpool2D_back(PoolDescriptor *D){
  _profile_fpga(_FPGA_AVGPOOL2D_BACK, 0);
#ifndef _K_ENABLED_AVGPOOL2D_BACK
  fpga_cpuemu_avgpool2D_back(D);
#else
  cl_int err;
  cl::Event event;

  // parameters
  int batch_size          = D->I->shape[0];     // batch size
  cl::Buffer ID           = *D->ID->fpga_ptr;   // input delta
  int IDrows              = D->ir;              // input rows
  int IDcols              = D->ic;              // input cols
  int IDchannels          = D->iz;              // input channels
  cl::Buffer D            = *D->O->fpga_ptr;    // D
  int padding_rows_top    = D->padrt;           // padding rows (top)
  int padding_rows_bottom = D->padrb;           // padding rows (bottom)
  int padding_cols_left   = D->padcl;           // padding cols (left)
  int padding_cols_right  = D->padcr;           // padding cols (right)
  int kernel_rows         = D->kr;              // kernel rows
  int kernel_cols         = D->kc;              // kernel cols
  int stride_rows         = D->sr;              // stride rows
  int stride_cols         = D->sc;              // stride cols
  int Dsize               = D->size;            // size (descriptor's field)

  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(0, batch_size));
  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(1, ID));
  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(2, IDrows));
  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(3, IDcols));
  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(4, IDchannels));
  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(5, D));
  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(6, padding_rows_top));
  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(7, padding_rows_bottom));
  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(8, padding_cols_left));
  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(9, padding_cols_right));
  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(10, kernel_rows));
  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(11, kernel_cols));
  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(12, stride_rows));
  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(13, stride_cols));
  OCL_CHECK(err, err = kernel_avgpool2D_back.setArg(14, Dsize));

  OCL_CHECK(err, err = q.enqueueTask(kernel_avgpool2D_back, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_AVGPOOL2D_BACK, 1);
}

#endif
