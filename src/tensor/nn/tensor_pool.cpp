/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/nn/gpu_tensor_nn.h"
#endif

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#endif

namespace tensorNN {


    void MPool2D(PoolDescriptor *D) {
        /////////////////////////////////////////////////////////////////////
        //// MPool2D
        //// Dimensions must be compatible
        //// A is input 4D Tensor, Batch x Channels x Rows x Cols
        //// D is a PoolDescriptor
        /////////////////////////////////////////////////////////////////////
        if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::MPool2D");

        D->O->tsem->lock();
        if (D->I->isCPU()) {
            cpu_mpool2D(D);
        }
#ifdef cGPU
        else if (D->I->isGPU())
          {
            gpu_mpool2D(D);
          }
#endif
#ifdef cFPGA
    else if (D->I->isFPGA())
      {
        fpga_mpool2D(D);
      }
#endif
        D->O->tsem->unlock();
    }

    void MPool2D_back(PoolDescriptor *D) {
        /////////////////////////////////////////////////////////////////////
        //// MPool2D
        //// Dimensions must be compatible
        //// A is input 4D Tensor, Batch x Channels x Rows x Cols
        //// D is a PoolDescriptor
        /////////////////////////////////////////////////////////////////////
        if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::MPool2D_back");

        D->ID->tsem->lock();
        if (D->I->isCPU()) {
            cpu_mpool2D_back(D);
        }
#ifdef cGPU
        else if (D->I->isGPU())
          {
            gpu_mpool2D_back(D);
          }
#endif
#ifdef cFPGA
    else if (D->I->isFPGA())
      {
        fpga_mpool2D_back(D);
      }
#endif
        D->ID->tsem->unlock();
    }


    void AvgPool2D(PoolDescriptor *D) {
        /////////////////////////////////////////////////////////////////////
        //// AvgPool2D
        //// Dimensions must be compatible
        //// A is input 4D Tensor, Batch x Channels x Rows x Cols
        //// D is a PoolDescriptor
        /////////////////////////////////////////////////////////////////////
        if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::AvgPool2D");

        D->O->tsem->lock();
        if (D->I->isCPU()) {
            cpu_avgpool2D(D);
        }
#ifdef cGPU
        else if (D->I->isGPU())
          {
            gpu_avgpool2D(D);
          }
#endif
#ifdef cFPGA
    else if (D->I->isFPGA())
      {
        fpga_avgpool2D(D);
      }
#endif
        D->O->tsem->unlock();
    }

    void AvgPool2D_back(PoolDescriptor *D) {
        /////////////////////////////////////////////////////////////////////
        //// AvgPool2D_back
        //// Dimensions must be compatible
        //// A is input 4D Tensor, Batch x Channels x Rows x Cols
        //// D is a PoolDescriptor
        /////////////////////////////////////////////////////////////////////
        if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::AvgPool2D_back");

        D->ID->tsem->lock();
        if (D->I->isCPU()) {
            cpu_avgpool2D_back(D);
        }
#ifdef cGPU
        else if (D->I->isGPU())
          {
            gpu_avgpool2D_back(D);
          }
#endif
#ifdef cFPGA
    else if (D->I->isFPGA())
      {
        fpga_avgpool2D_back(D);
      }
#endif
        D->ID->tsem->unlock();
    }

}