/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cmath>
#include <limits>
#include <iostream>
#include <utility>

#include "eddl/tensor/tensor.h"
#include "eddl/hardware/cpu/cpu_tensor.h"
#include "eddl/profiling.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#endif

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#endif

using namespace std;


Tensor* Tensor::shift(vector<int> shift, WrappingMode mode, float cval){
    Tensor *t_new = Tensor::empty_like(this);
    Tensor::shift(this, t_new, std::move(shift), mode, cval);
    return t_new;
}

void Tensor::shift(Tensor *A, Tensor *B, vector<int> shift, WrappingMode mode, float cval){
    // shift => {y, x}
    // Parameter check
    if(::abs(shift[0]) >= A->shape[2] || ::abs(shift[1]) >= A->shape[3]){
        msg("The shift is greater than the image size", "Tensor::shift");
    }

    // Check dimensions
    if(A->shape!=B->shape){
        msg("Incompatible dimensions", "Tensor::shift");
    } else if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::shift");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::shift");
    }

    PROFILING_HEADER_EXTERN(shift);

    if (A->isCPU()) {
        cpu_shift(A, B, std::move(shift), mode, cval);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_shift(A, B, std::move(shift), mode, cval);
      }
#endif
#ifdef cFPGA
    else {
        fpga_shift(A, B, std::move(shift), mode, cval);
    }
#endif

    PROFILING_FOOTER(shift);
}

Tensor* Tensor::rotate(float angle, vector<int> offset_center, WrappingMode mode, float cval){
    Tensor *t_new = Tensor::empty_like(this);
    Tensor::rotate(this, t_new, angle, std::move(offset_center), mode, cval);
    return t_new;
}

void Tensor::rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center, WrappingMode mode, float cval) {
    // Check dimensions
    if(A->shape!=B->shape){
        msg("Incompatible dimensions", "Tensor::rotate");
    } else if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::rotate");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::rotate");
    }

    PROFILING_HEADER_EXTERN(rotate);

    if (A->isCPU()) {
        cpu_rotate(A, B, angle, std::move(offset_center), mode, cval);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_rotate(A, B, angle, std::move(offset_center), mode, cval);
      }
#endif
#ifdef cFPGA
    else {
        fpga_rotate(A, B, angle, std::move(offset_center), mode, cval);
    }
#endif

    PROFILING_FOOTER(rotate);
}

Tensor* Tensor::scale(vector<int> new_shape, WrappingMode mode, float cval, bool keep_size) {
    Tensor *t_new;

    if(keep_size){
        t_new = Tensor::empty_like(this);
    }else{
        int height = new_shape[0];
        int width = new_shape[1];
        t_new = Tensor::empty({this->shape[0], this->shape[1], height, width});
    }
    Tensor::scale(this, t_new, new_shape, mode, cval);
    return t_new;
}

void Tensor::scale(Tensor *A, Tensor *B, vector<int> new_shape, WrappingMode mode, float cval) {
    // new_shape => {y, x}
    // Parameter check
    if(new_shape[0] <= 0 || new_shape[1] <= 0){
        msg("The new shape must be a greater than zero", "Tensor::scale");
    }

    // Check dimensions
    if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::scale");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::scale");
    }

    PROFILING_HEADER_EXTERN(scale);

    if (A->isCPU()) {
        cpu_scale(A, B, std::move(new_shape), mode, cval);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_scale(A, B, std::move(new_shape), mode, cval);
      }
#endif
#ifdef cFPGA
    else {
        fpga_scale(A, B, std::move(new_shape), mode, cval);
    }
#endif

    PROFILING_FOOTER(scale);
}


Tensor* Tensor::flip(int axis) {
    Tensor *t_new = Tensor::empty_like(this);
    Tensor::flip(this, t_new, axis);
    return t_new;
}

void Tensor::flip(Tensor *A, Tensor *B, int axis) {
    // Parameter check
    if(axis != 0 && axis != 1){
        msg("Axis must be either 0 (vertical axis) or 1 (horizontal axis)", "Tensor::flip");
    }

    // Check dimensions
    if(A->shape!=B->shape){
        msg("Incompatible dimensions", "Tensor::flip");
    } else if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::flip");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::flip");
    }

    PROFILING_HEADER_EXTERN(flip);

    if (A->isCPU()) {
        cpu_flip(A, B, axis);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_flip(A, B, axis);
      }
#endif
#ifdef cFPGA
    else {
        fpga_flip(A, B, axis);
    }
#endif

    PROFILING_FOOTER(flip);
}

Tensor* Tensor::crop(vector<int> coords_from, vector<int> coords_to, float cval, bool keep_size){
    Tensor *t_new;
    if(keep_size){
        t_new = Tensor::empty_like(this);
    }else{
        int height = coords_to[0]-coords_from[0]+1;
        int width = coords_to[1]-coords_from[1]+1;
        t_new = Tensor::empty({this->shape[0], this->shape[1], height, width});
    }
    Tensor::crop(this, t_new, coords_from, coords_to, cval);
    return t_new;
}

void Tensor::crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float cval) {
    // coords => {y, x}
    // Parameter check
    if(coords_from[0] < 0 || coords_from[0]>= A->shape[2] ||
       coords_from[1] < 0 || coords_from[1]>= A->shape[3] ||
       coords_to[0] < 0 || coords_to[0]>= A->shape[2] ||
       coords_to[1] < 0 || coords_to[1]>= A->shape[3]){
        msg("Crop coordinates must fall within the range of the tensor", "Tensor::crop");
    }

    // Check dimensions
    if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::crop");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::crop");
    }

    PROFILING_HEADER_EXTERN(crop);

    if (A->isCPU()) {
        cpu_crop(A, B, std::move(coords_from), std::move(coords_to), cval, false);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_crop(A, B, std::move(coords_from), std::move(coords_to), cval, false);
      }
#endif
#ifdef cFPGA
    else {
        fpga_crop(A, B, std::move(coords_from), std::move(coords_to), cval, false);
    }
#endif

    PROFILING_FOOTER(crop);
}

Tensor* Tensor::crop_scale(vector<int> coords_from, vector<int> coords_to, WrappingMode mode, float cval){
    Tensor *t_new = Tensor::empty_like(this);
    Tensor::crop_scale(this, t_new, coords_from, coords_to, mode, cval);
    return t_new;
}

void Tensor::crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, WrappingMode mode, float cval) {
    // coords => {y, x}
    // Parameter check
    if(coords_from[0] < 0 || coords_from[0]>= A->shape[2] ||
       coords_from[1] < 0 || coords_from[1]>= A->shape[3] ||
       coords_to[0] < 0 || coords_to[0]>= A->shape[2] ||
       coords_to[1] < 0 || coords_to[1]>= A->shape[3]){
       msg("Crop coordinates must fall within the range of the tensor", "Tensor::crop_scale");
    }

    // Check dimensions
    if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::crop_scale");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::crop_scale");
    }

    PROFILING_HEADER_EXTERN(crop_scale);

    if (A->isCPU()) {
        cpu_crop_scale(A, B, std::move(coords_from), std::move(coords_to), mode, cval);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_crop_scale(A, B, std::move(coords_from), std::move(coords_to), mode, cval);
      }
#endif
#ifdef cFPGA
    else {
        fpga_crop_scale(A, B, std::move(coords_from), std::move(coords_to), mode, cval);
    }
#endif

    PROFILING_FOOTER(crop_scale);
}


Tensor* Tensor::cutout(vector<int> coords_from, vector<int> coords_to, float cval) {
    Tensor *t_new = Tensor::empty_like(this);
    Tensor::cutout(this, t_new, coords_from, coords_to, cval);
    return t_new;
}

void Tensor::cutout(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float cval) {
    // coords => {y, x}
    // Parameter check
    if(coords_from[0] < 0 || coords_from[0]>= A->shape[2] ||
       coords_from[1] < 0 || coords_from[1]>= A->shape[3] ||
       coords_to[0] < 0 || coords_to[0]>= A->shape[2] ||
       coords_to[1] < 0 || coords_to[1]>= A->shape[3]){
       msg("Cutout coordinates must fall within the range of the tensor", "Tensor::cutout");
    }

    // Check dimensions
    if(A->shape!=B->shape){
        msg("Incompatible dimensions", "Tensor::cutout");
    } else if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::cutout");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::cutout");
    }

    PROFILING_HEADER_EXTERN(cutout);

    if (A->isCPU()) {
        cpu_crop(A, B, std::move(coords_from), std::move(coords_to), cval, true);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_crop(A, B, std::move(coords_from), std::move(coords_to), cval, true);
      }
#endif
#ifdef cFPGA
    else {
        fpga_crop(A, B, std::move(coords_from), std::move(coords_to), cval, true);
    }
#endif

    PROFILING_FOOTER(cutout);
}

Tensor* Tensor::pad(vector<int> pads, float cval) {
    // Parameter check
    if(pads.size()==2){
        pads = vector<int>({pads[0], pads[1], pads[0], pads[1]});
    } else if(pads.size()==4){ }
    else{
        msg("The padding on each border must follow this format (top-bottom, left-right) or (top, right, bottom, left)", "Tensor::pad");
    }

    Tensor* t_new = Tensor::full({this->shape[0], this->shape[1], this->shape[2]+(pads[0]+pads[2]), this->shape[3]+(pads[1]+pads[3])}, cval, this->device);
    Tensor::pad(this, t_new, pads);
    return t_new;
}

void Tensor::pad(Tensor *A, Tensor *B, vector<int> pads) {
    // Parameter check
    if(pads.size()==2){
        pads = vector<int>({pads[0], pads[1], pads[0], pads[1]});
    } else if(pads.size()==4){ }
    else{
        msg("The padding on each border must follow this format (top-bottom, left-right) or (top, right, bottom, left)", "Tensor::pad");
    }

    if(pads[0] < 0 || pads[1] < 0  || pads[2] < 0 || pads[3] < 0){
        msg("Pad margin must be greater or equal than zero", "Tensor::pad");
    }

    // Check dimensions
    if(A->shape[0]!=B->shape[0] || A->shape[1]!=B->shape[1] ||
        (A->shape[2]+(pads[0]+pads[2]))!=B->shape[2] || (A->shape[3]+(pads[1]+pads[3])!=B->shape[3])){
        msg("Incompatible dimensions", "Tensor::pad");
    } else if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::pad");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::pad");
    }

//    PROFILING_HEADER_EXTERN(pad);

    if (A->isCPU()){
        cpu_pad(A, B, std::move(pads));
    }
#ifdef cGPU
    else if (A->isGPU())
    {
        gpu_pad(A, B, std::move(pads));
    }
#endif
#ifdef cFPGA
        else {
//        fpga_pad(A, B, std::move(pads));
    }
#endif

//    PROFILING_FOOTER(pad);
}


void Tensor::pad_back(Tensor *A, Tensor *B, vector<int> pads){
    // Parameter check
    if(pads.size()==2){
        pads = vector<int>({pads[0], pads[1], pads[0], pads[1]});
    } else if(pads.size()==4){ }
    else{
        msg("The padding on each border must follow this format (top-bottom, left-right) or (top, right, bottom, left)", "Tensor::pad");
    }

    if(pads[0] < 0 || pads[1] < 0  || pads[2] < 0 || pads[3] < 0){
        msg("Pad margin must be greater or equal than zero", "Tensor::pad");
    }

    // Check dimensions
    if(A->shape[0]!=B->shape[0] || A->shape[1]!=B->shape[1] ||
       A->shape[2]!=(B->shape[2]-(pads[0]+pads[2])) || A->shape[3]!=(B->shape[3]-(pads[1]+pads[3]))){
        msg("Incompatible dimensions", "Tensor::pad_back");
    } else if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::pad_back");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::pad_back");
    }

//    PROFILING_HEADER_EXTERN(pad_back);

    if (A->isCPU()){
        cpu_pad_back(A, B, std::move(pads));
    }
#ifdef cGPU
    else if (A->isGPU())
    {
        gpu_pad_back(A, B, std::move(pads));
    }
#endif
#ifdef cFPGA
    else {
//        fpga_pad_back(A, B, std::move(pads));
    }
#endif

//    PROFILING_FOOTER(pad_back);
}

Tensor* Tensor::shift_random(vector<float> factor_x, vector<float> factor_y, WrappingMode mode, float cval){
    Tensor *t_new = Tensor::empty_like(this);
    Tensor::shift_random(this, t_new, factor_x, factor_y, mode, cval);
    return t_new;
}


void Tensor::shift_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, WrappingMode mode, float cval){
    // Parameter check
    if(factor_x[0] < -1.0f || factor_x[0] > 1.0f ||
       factor_x[1] < -1.0f || factor_x[1] > 1.0f ||
       factor_y[0] < -1.0f || factor_y[0] > 1.0f ||
       factor_y[1] < -1.0f || factor_y[1] > 1.0f){
        msg("The shift factors must fall within the range [-1.0, 1.0]", "Tensor::shift_random");
    }

    // Check dimensions
    if(A->shape!=B->shape){
        msg("Incompatible dimensions", "Tensor::shift_random");
    } else if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::shift_random");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::shift_random");
    }

    PROFILING_HEADER_EXTERN(shift_random);

    if (A->isCPU()) {
        cpu_shift_random(A, B, std::move(factor_x), std::move(factor_y), mode, cval);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_shift_random(A, B, std::move(factor_x), std::move(factor_y), mode, cval);
      }
#endif
#ifdef cFPGA
    else {
        fpga_shift_random(A, B, std::move(factor_x), std::move(factor_y), mode, cval);
    }
#endif

    PROFILING_FOOTER(shift_random);
}


Tensor* Tensor::rotate_random(vector<float> factor, vector<int> offset_center, WrappingMode mode, float cval){
    Tensor *t_new = Tensor::empty_like(this);
    Tensor::rotate_random(this, t_new, factor, offset_center, mode, cval);
    return t_new;
}

void Tensor::rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center, WrappingMode mode, float cval) {
    // Check dimensions
    if(A->shape!=B->shape){
        msg("Incompatible dimensions", "Tensor::rotate_random");
    } else if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::rotate_random");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::rotate_random");
    }

    PROFILING_HEADER_EXTERN(rotate_random);

    if (A->isCPU()) {
        cpu_rotate_random(A, B,  std::move(factor), std::move(offset_center), mode, cval);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_rotate_random(A, B,  std::move(factor), std::move(offset_center), mode, cval);
      }
#endif
#ifdef cFPGA
    else {
        fpga_rotate_random(A, B,  std::move(factor), std::move(offset_center), mode, cval);
    }
#endif

    PROFILING_FOOTER(rotate_random);
}

Tensor* Tensor::scale_random(vector<float> factor, WrappingMode mode, float cval){
    // We don't accept keep_size!

    Tensor *t_new = Tensor::empty_like(this);
    Tensor::scale_random(this, t_new, factor, mode, cval);
    return t_new;
}

void Tensor::scale_random(Tensor *A, Tensor *B, vector<float> factor, WrappingMode mode, float cval) {
    // Parameter check
    if(factor[0] < 0.0f || factor[1] < 0.0f){
        msg("The scaling factor must be a positive number", "Tensor::scale_random");
    }

    // Check dimensions
    if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::scale_random");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::scale_random");
    }

    PROFILING_HEADER_EXTERN(scale_random);

    if (A->isCPU()) {
        cpu_scale_random(A, B, std::move(factor), mode, cval);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_scale_random(A, B, std::move(factor), mode, cval);
      }
#endif
#ifdef cFPGA
    else {
        fpga_scale_random(A, B, std::move(factor), mode, cval);
    }
#endif

    PROFILING_FOOTER(scale_random);
}


Tensor* Tensor::flip_random(int axis){
    Tensor *t_new = Tensor::empty_like(this);
    Tensor::flip_random(this, t_new, axis);
    return t_new;
}

void Tensor::flip_random(Tensor *A, Tensor *B, int axis) {
    // Parameter check
    if(axis != 0 && axis != 1){
        msg("The axis must be either 0 (vertical axis) or 1 (horizontal axis)", "Tensor::flip_random");
    }

    // Check dimensions
    if(A->shape!=B->shape){
        msg("Incompatible dimensions", "Tensor::flip_random");
    } else if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::flip_random");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::flip_random");
    }

    PROFILING_HEADER_EXTERN(flip_random);

    if (A->isCPU()) {
        cpu_flip_random(A, B, axis);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_flip_random(A, B, axis);
      }
#endif
#ifdef cFPGA
    else {
        fpga_flip_random(A, B, axis);
    }
#endif

    PROFILING_FOOTER(flip_random);
}

Tensor* Tensor::crop_random(int height, int width, float cval, bool keep_size){
    // Check height and width
    if(height <= 0 || height > this->shape[2]){
        msg("The height must be smaller than the current tensor height and greater than zero", "Tensor::crop_random");
    }
    if(width <= 0 || width > this->shape[3]){
        msg("The width must be smaller than the current tensor width and greater than zero", "Tensor::crop_random");
    }

    // Perform crop
    Tensor *t_new;
    if(keep_size){
        t_new = Tensor::full(this->shape, cval);
        // Canvas => Paste patch
        msg("'keep_size=true' not yet implemented", "Tensor::crop_random");
    }else{
        t_new = Tensor::full({this->shape[0], this->shape[1], height, width}, cval);
        Tensor::crop_random(this, t_new);  // Get patch => Copy
    }
    return t_new;
}

void Tensor::crop_random(Tensor *A, Tensor *B) {
    // Check dimensions
    if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::crop_random");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::crop_random");
    }

    PROFILING_HEADER_EXTERN(crop_random);

    if (A->isCPU()) {
        cpu_crop_random(A, B);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_crop_random(A, B);
      }
#endif
#ifdef cFPGA
    else {
        fpga_crop_random(A, B);
    }
#endif

    PROFILING_FOOTER(crop_random);
}

Tensor* Tensor::crop_scale_random(vector<float> factor, WrappingMode mode, float cval){
    Tensor *t_new = Tensor::empty_like(this);
    Tensor::crop_scale_random(this, t_new, factor, mode, cval);
    return t_new;
}

void Tensor::crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, WrappingMode mode, float cval) {
    // Parameter check
    if(factor[0] < 0.0f || factor[0] > 1.0f ||
       factor[1] < 0.0f || factor[1] > 1.0f){
       msg("The crop factor must fall within the range [0.0, 1.0]", "Tensor::crop_scale_random");
    }

    // Check dimensions
    if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::crop_scale_random");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::crop_scale_random");
    }

    PROFILING_HEADER_EXTERN(crop_scale_random);

    if (A->isCPU()) {
        cpu_crop_scale_random(A, B, std::move(factor), mode, cval);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_crop_scale_random(A, B, std::move(factor), mode, cval);
      }
#endif
#ifdef cFPGA
    else {
        fpga_crop_scale_random(A, B, std::move(factor), mode, cval);
    }
#endif

    PROFILING_FOOTER(crop_scale_random);
}

Tensor* Tensor::cutout_random(vector<float> factor_x, vector<float> factor_y, float cval){
    Tensor *t_new = Tensor::empty_like(this);
    Tensor::cutout_random(this, t_new, factor_x, factor_y, cval);
    return t_new;
}

void Tensor::cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float cval) {
    // Parameter check
    if(factor_x[0] < 0.0f || factor_x[0] > 1.0f ||
       factor_x[1] < 0.0f || factor_x[1] > 1.0f ||
       factor_y[0] < 0.0f || factor_y[0] > 1.0f ||
       factor_y[1] < 0.0f || factor_y[1] > 1.0f){
       msg("The cutout factors must fall within the range [0.0, 1.0]", "Tensor::cutout_random");
    }

    // Check dimensions
    if(A->shape!=B->shape){
        msg("Incompatible dimensions", "Tensor::cutout_random");
    } else if (A->ndim != 4 || B->ndim != 4){
        msg("This method requires two 4D tensors", "Tensor::cutout_random");
    } else if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::cutout_random");
    }

    PROFILING_HEADER_EXTERN(cutout_random);

    if (A->isCPU()) {
        cpu_cutout_random(A, B, std::move(factor_x), std::move(factor_y), cval);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_cutout_random(A, B, std::move(factor_x), std::move(factor_y), cval);
      }
#endif
#ifdef cFPGA
    else {
        fpga_cutout_random(A, B, std::move(factor_x), std::move(factor_y), cval);
    }
#endif

  PROFILING_FOOTER(cutout_random);
}
