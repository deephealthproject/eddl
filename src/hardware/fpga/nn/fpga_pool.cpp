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

// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_mpool2D         = 1;
char fpga_set_cpuemu_mpool2D_back    = 1;
char fpga_set_cpuemu_avgpool2D       = 1;
char fpga_set_cpuemu_avgpool2D_back  = 1;

fpga_data_type fpga_get_pixel(int b,int px,int py,int pz,PoolDescriptor *D,int isize,int irsize) {

  fpga_data_type *ptr = (fpga_data_type *)D->I->ptr;
  // Check boundaries of the window
  if (px<0) return fpga_data_type(0.0);
  if (py<0) return fpga_data_type(0.0);
  if (px>=D->ic) return fpga_data_type(0.0);
  if (py>=D->ir) return fpga_data_type(0.0);

  // Compute address from indices (row-major)
  unsigned int address = (b*isize) + (pz*irsize) + (py*D->ic) + px;
  return ptr[address];
}

// -----------------------------------------------------------------
// mpool2D
//
void fpga_cpuemu_mpool2D(PoolDescriptor *D) {

  fpga_copy_from_fpga(D->I, D->I->ptr, 0);

  int isize = D->ir*D->ic*D->iz;
  int irsize = D->ir*D->ic;

  #pragma omp parallel for default(none) shared(D, isize, irsize)
  for(int b=0; b<D->I->shape[0]; b++){  // Batches
      int p=b*D->size;  // Kernel's index (opt. shared variable)

      for(int k=0; k<D->iz; k++) { // Depth: front-back
          for(int i=-D->padrt; i<=D->ir+D->padrb-D->kr; i+=D->sr) {  // rows: top-bottom
              for(int j=-D->padcl; j<=D->ic+D->padcr-D->kc; j+=D->sc, p++) { // cols: left-right

                  // Get max value in window
                  fpga_data_type max = CPU_LOWEST_FLOAT;
                  for(int ki=0; ki<D->kr; ki++){  // rows (kernel): top-bottom
                      for(int kj=0; kj<D->kc; kj++) { // cols (kernel): left-right

                          // Get value W[ki,kj] value in window
                          float v = fpga_get_pixel(b,j+kj,i+ki, k, D, isize, irsize);
                          if (v>max) {
                              max = v;
                              D->indX->ptr[p] = j+kj;
                              D->indY->ptr[p] = i+ki;
                          }

                      } // kernel cols
                  }  // kernel rows

                  // Set output value
                  fpga_data_type *ptr = (fpga_data_type *)D->O->ptr;
                  ptr[p] = max;

              } // cols
          } // rows
      } // depth
  } // batch

  fpga_copy_to_fpga(D->O->ptr, D->O, 0);
}

void fpga_mpool2D(PoolDescriptor *D){
    _debug_fpga_funcs("mpool2D");
    _profile_fpga(_FPGA_MPOOL2D, 0);
    _profile_fpga_tensor(D->I);
    _profile_fpga_tensor(D->O);
    if (fpga_set_cpuemu_mpool2D == 1) {
        fpga_cpuemu_mpool2D(D);
    } else {
        cl_int err;
        cl::Event event;

    /*    OCL_CHECK(err, err = kernel_mpool2D.setArg(0, (int)D->ir));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(1, (int)D->ic));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(2, (int)D->iz));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(3, (int)D->padrt));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(4, (int)D->padrb));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(5, (int)D->padcl));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(6, (int)D->padcr));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(7, (int)D->kr));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(8, (int)D->kc));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(9, (int)D->sr));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(10, (int)D->sc));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(11, (long int)D->size));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(12, (int)D->I->shape[0]));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(13, *((cl::Buffer*)D->I->fpga_ptr)));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(14, *((cl::Buffer*)D->O->fpga_ptr)));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(15, *((cl::Buffer*)D->indX->fpga_ptr)));
        OCL_CHECK(err, err = kernel_mpool2D.setArg(16, *((cl::Buffer*)D->indY->fpga_ptr)));

        OCL_CHECK(err, err = q.enqueueTask(kernel_mpool2D, NULL, &event));
        q.finish();*/
    }
    _profile_fpga_tensor(D->O);
    _profile_fpga(_FPGA_MPOOL2D, 1);
}

// -----------------------------------------------------------------
// mpool2D_back
//
void fpga_cpuemu_mpool2D_back(PoolDescriptor *D) {
  fpga_copy_from_fpga(D->D, D->D->ptr);
  cpu_mpool2D_back(D);
  fpga_copy_to_fpga(D->ID->ptr, D->ID);
}

void fpga_mpool2D_back(PoolDescriptor *D){
    _profile_fpga(_FPGA_MPOOL2D_BACK, 0);
    if (fpga_set_cpuemu_mpool2D_back == 1) {
        fpga_cpuemu_mpool2D_back(D);
    } else {
      cl_int err;
      cl::Event event;

     /* OCL_CHECK(err, err = kernel_mpool2D_back.setArg(0, (int)D->ir));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(1, (int)D->ic));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(2, (int)D->iz));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(3, (int)D->padrt));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(4, (int)D->padrb));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(5, (int)D->padcl));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(6, (int)D->padcr));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(7, (int)D->kr));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(8, (int)D->kc));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(9, (int)D->sr));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(10, (int)D->sc));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(11, (long int)D->size));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(12, (int)D->I->shape[0]));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(13, *((cl::Buffer*)D->I->fpga_ptr)));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(14, *((cl::Buffer*)D->O->fpga_ptr)));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(15, *((cl::Buffer*)D->indX->fpga_ptr)));
      OCL_CHECK(err, err = kernel_mpool2D_back.setArg(16, *((cl::Buffer*)D->indY->fpga_ptr)));

      OCL_CHECK(err, err = q.enqueueTask(kernel_mpool2D, NULL, &event));
      q.finish();*/
    }
    _profile_fpga(_FPGA_MPOOL2D_BACK, 1);
}

// -----------------------------------------------------------------
// avgpool2D
//
void fpga_cpuemu_avgpool2D(PoolDescriptor *D) {
    printf("fpga_cpuemu_avgpool2D not implemented yet\n");
    exit(1);
}

void fpga_avgpool2D(PoolDescriptor *D){
    _profile_fpga(_FPGA_AVGPOOL2D, 0);
    if (fpga_set_cpuemu_avgpool2D == 1) {
        fpga_cpuemu_avgpool2D(D);
    } else {
        printf("fpga_avgpool2D not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_AVGPOOL2D, 1);
}

// -----------------------------------------------------------------
// avgpool2D_back
//
void fpga_cpuemu_avgpool2D_back(PoolDescriptor *D) {
    printf("fpga_cpuemu_avgpool2D_back not implemented yet\n");
    exit(1);
}

void fpga_avgpool2D_back(PoolDescriptor *D){
    _profile_fpga(_FPGA_AVGPOOL2D_BACK, 0);
    if (fpga_set_cpuemu_avgpool2D_back == 1) {
        fpga_cpuemu_avgpool2D_back(D);
    } else {
        printf("fpga_avgpool2D_back not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_AVGPOOL2D_BACK, 1);
}

#endif
