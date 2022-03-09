/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: December 2021
* Author: GAP Research Group (UPV), contact: jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

#include <cstdio>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <math.h>

#include "eddl/apis/eddl.h"
#include "eddl/utils.h"
#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/cpu/cpu_tensor.h"

#define ENABLE_UPSIZE_SUPPORT
#define ENABLE_CLAMP

////////////////////////////////////////////////////////
///// EDDL is a wrapper class to ease and define the API
////////////////////////////////////////////////////////

using namespace std;

namespace eddl {

extern void fpga_reshape_kernel(ConvolDescriptor *src_D, ConvolDescriptor *D, int KW, int KH, int I, int O, int CPI, int CPO);
extern void _profile_fpga_tensor(char *str, Tensor *t, int format_tensor);

#define MAX_ASSOCIATED_LAYERS 1000
struct {
  Layer *src;
  Layer *dst_ghwc;    // dst layer in ghwc format
  Layer *dst_chw;     // dst layer in chw format
  int   layer_id_ghwc;
  int   layer_id_chw;
  int   format;
  int   device;
} associated_layers[MAX_ASSOCIATED_LAYERS];

struct {
  void* src;
  void* conv;
  void* bn;
  void* dense;
  void* dst;
} associated_source_layer[MAX_ASSOCIATED_LAYERS];

#define chw_format 0
#define ghwcpi_format 1
#define cpu_device 0
#define fpga_device 1

int current_associated_layers = 0;

  void fn_set_associated_layer(Layer *src, Layer *dst, int ghwc_format, int format, int device, int layer_id) {
    //printf("set associated: src %p dst %p, ghwc_format %d layer id %d\n", src, dst, ghwc_format, layer_id);
    // let's find the entry
    int found = 0;
    int i;
    for (i=0; i<current_associated_layers; i++) {
      if (associated_layers[i].src == src) {
        found = 1;
        break;
      }
    }
    if (!found) {
      i = current_associated_layers;
      associated_layers[i].src = src;
      associated_layers[i].dst_ghwc = NULL;
      associated_layers[i].dst_chw = NULL;
      associated_layers[i].layer_id_ghwc = -1;
      associated_layers[i].layer_id_chw = -1;
      associated_layers[i].format = -1;
      associated_layers[i].device = -1;
      current_associated_layers++;
    }

    if (ghwc_format) {
      associated_layers[i].dst_ghwc = dst; 
      associated_layers[i].layer_id_ghwc = layer_id;
    } else {
      associated_layers[i].dst_chw = dst;
      associated_layers[i].layer_id_chw = layer_id;
    }
    associated_layers[i].format = format;
    associated_layers[i].device = device;
  }

  Layer *fn_get_associated_layer(Layer *src, int ghwc_format, int *layer_id) {
    //printf("getting associated layer from %p in format %d\n", src, ghwc_format);
    for (int i=0; i<current_associated_layers; i++) {
      if (associated_layers[i].src == src) {
        if (ghwc_format) {
          *layer_id = associated_layers[i].layer_id_ghwc;
          return associated_layers[i].dst_ghwc;
        } else {
          *layer_id = associated_layers[i].layer_id_chw;
          return associated_layers[i].dst_chw;
        }
      }
    }
    msg("Error, associated layer not found","fn_get_associated_layer");
    return NULL;
  }

  int fn_get_associated_layer_format(Layer *src) {
    for (int i=0; i<current_associated_layers; i++) {
      if (associated_layers[i].src == src) return associated_layers[i].format;
    }
    msg("Error, associated layer not found","fn_get_associated_layer_format");
    return -1;
  }

  int fn_get_associated_layer_device(Layer *src) {
    for (int i=0; i<current_associated_layers; i++) {
      if (associated_layers[i].src == src) return associated_layers[i].device;
    }
    msg("Error, associated layer not found","fn_get_associated_layer_device");
    return -1;
  }

  void* fn_get_cpu_equivalent_layer(void* dst, int n_associated_layers) {
    for (int i=0; i<n_associated_layers; i++) {
      if (associated_source_layer[i].dst == dst) {
        return associated_source_layer[i].src;
      }
    }
    msg("Error, associated layer not found", "fn_get_cpu_equivalent_layer");
    return NULL;
 }

void* fn_get_cpu_equivalent_conv_layer(void* dst, int n_associated_layers) {
    for (int i=0; i<n_associated_layers; i++) {
      if (associated_source_layer[i].dst == dst) {
        return associated_source_layer[i].conv;
      }
    }
    msg("Error, associated layer not found", "fn_get_cpu_equivalent_conv_layer");
    return NULL;
  }

void* fn_get_cpu_equivalent_bn_layer(void* dst, int n_associated_layers) {
    for (int i=0; i<n_associated_layers; i++) {
      if (associated_source_layer[i].dst == dst) {
        return associated_source_layer[i].bn;
      }
    }
    msg("Error, associated layer not found", "fn_get_cpu_equivalent_bn_layer");
    return NULL;
  }

void* fn_get_cpu_equivalent_dense_layer(void* dst, int n_associated_layers) {
    for (int i=0; i<n_associated_layers; i++) {
      if (associated_source_layer[i].dst == dst) {
        return associated_source_layer[i].dense;
      }
    }
    msg("Error, associated layer not found", "fn_get_cpu_equivalent_bn_layer");
    return NULL;
}

// Single layer identification
bool is_input       (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LInput *dl = dynamic_cast<LInput *>(cl)) return true; return false;}
bool is_relu        (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LActivation *dl = dynamic_cast<LActivation *>(cl)) if (dl->act == "relu") return true; return false;}
bool is_leakyrelu   (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LActivation *dl = dynamic_cast<LActivation *>(cl)) if (dl->act == "leaky_relu") return true; return false;}      
bool is_softmax     (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LActivation *dl = dynamic_cast<LActivation *>(cl)) if (dl->act == "softmax") return true; return false;}
bool is_sigmoid     (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LActivation *dl = dynamic_cast<LActivation *>(cl)) if (dl->act == "sigmoid") return true; return false;}
bool is_softplus    (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LActivation *dl = dynamic_cast<LActivation *>(cl)) if (dl->act == "softplus") return true; return false;}
bool is_tanh        (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LActivation *dl = dynamic_cast<LActivation *>(cl)) if (dl->act == "tanh") return true; return false;}
bool is_linear      (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LActivation *dl = dynamic_cast<LActivation *>(cl)) if (dl->act == "linear") return true; return false;}
bool is_maxp        (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LMaxPool *dl = dynamic_cast<LMaxPool *>(cl)) return true; return false;}
bool is_avgp        (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LAveragePool *dl = dynamic_cast<LAveragePool *>(cl)) return true; return false;}
bool is_reshape     (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LReshape *dl = dynamic_cast<LReshape *>(cl)) return true; return false;}
bool is_resize      (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LResize *dl = dynamic_cast<LResize *>(cl)) return true; return false;}
bool is_dense       (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LDense *dl = dynamic_cast<LDense *>(cl)) return true; return false;}
bool is_concat      (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LConcat *dl = dynamic_cast<LConcat *>(cl)) return true; return false;}
bool is_expand      (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LExpand *dl = dynamic_cast<LExpand *>(cl)) return true; return false;}
bool is_select      (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LSelect *dl = dynamic_cast<LSelect *>(cl)) return true; return false;}
bool is_mult        (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LMult *dl = dynamic_cast<LMult *>(cl)) return true; return false;}
bool is_div         (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LDiv *dl = dynamic_cast<LDiv *>(cl)) return true; return false;}
bool is_diff        (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LDiff *dl = dynamic_cast<LDiff *>(cl)) return true; return false;}
bool is_exp         (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LExp *dl = dynamic_cast<LExp *>(cl)) return true; return false;}
bool is_permute     (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LPermute *dl = dynamic_cast<LPermute *>(cl)) return true; return false;}
bool is_add         (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LAdd *dl = dynamic_cast<LAdd *>(cl)) return true; return false;}
bool is_constoft    (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LConstOfTensor *dl = dynamic_cast<LConstOfTensor *>(cl)) return true; return false;}
bool is_clamp       (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LClamp *dl = dynamic_cast<LClamp *>(cl)) return true; return false;}
bool is_pad         (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LPad *dl = dynamic_cast<LPad *>(cl)) return true; return false;}
bool is_upsampling  (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LUpSampling *dl = dynamic_cast<LUpSampling *>(cl)) return true; return false;}
bool is_bn          (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LBatchNorm *dl = dynamic_cast<LBatchNorm *>(cl)) return true; return false;}
bool is_dropout     (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LDropout *dl = dynamic_cast<LDropout *>(cl)) return true; return false;}

bool is_conv        (model m_src, int layer, int num_layers) {if (layer >= num_layers) return false; Layer *cl = m_src->layers[layer]; if (LConv *dl = dynamic_cast<LConv *>(cl)) return true; return false;}


bool is_conv_fpga(model m, int l, int nl) {
  if (is_conv(m, l, nl)) {
    LConv *cl = (LConv *)m->layers[l];
    int found_C_cpu = 0;
    // only W dimensios less or equal than wmax are suported in the FPGA
    if (cl->input->shape.size() > 2) {
      if (cl->input->shape[3] > hlsinf_wo_max) found_C_cpu = 1;
    }
    // only filters of size 3 x 3 maximum
    if (cl->cd->kr > 3 || cl->cd->kc > 3) found_C_cpu = 1;
    if (!found_C_cpu) {return true;}
  } 
  return false;
}

bool found_dense_relu(model m, int l, int nl)           {return hlsinf_dense_support && is_dense(m, l, nl) && is_relu(m, l+1, nl);}
bool found_dense_div_clamp_relu(model m, int l, int nl) {return hlsinf_dense_support && is_dense(m, l, nl) && is_div(m, l+1, nl) && is_clamp(m, l+2, nl) && is_relu(m, l+3, nl);}
bool found_dense_div_clamp(model m, int l, int nl)      {return hlsinf_dense_support && is_dense(m, l, nl) && is_div(m, l+1, nl) && is_clamp(m, l+2, nl) && !found_dense_div_clamp_relu(m, l, nl);}
bool found_dense_hlsinf(model m, int l, int nl)         {return hlsinf_dense_support && is_dense(m, l, nl) && !found_dense_relu(m, l, nl) && !found_dense_div_clamp(m, l, nl) && !found_dense_div_clamp_relu(m, l, nl);}


bool is_conv_cpu(model m, int l, int nl) {return (is_conv(m, l, nl) && !is_conv_fpga(m, l, nl));}

// Layers not supported on HLSinf (these will run on cpu)
bool found_conv_cpu(model m, int l, int nl)        {return is_conv_cpu(m, l, nl);}
bool found_input(model m, int l, int nl)           {return is_input(m, l, nl);}
bool found_concat(model m, int l, int nl)          {return is_concat(m, l, nl);}
bool found_maxp(model m, int l, int nl)            {return is_maxp(m, l, nl);}
bool found_add(model m, int l, int nl)             {return is_add(m, l, nl);}
bool found_upsampling(model m, int l, int nl)      {return is_upsampling(m, l, nl);}
bool found_linear(model m, int l, int nl)          {return is_linear(m, l, nl);}
bool found_relu(model m, int l, int nl)            {return is_relu(m, l, nl);}
bool found_bn(model m, int l, int nl)              {return is_bn(m, l, nl);}
bool found_avgp(model m, int l, int nl)            {return is_avgp(m, l, nl);}
bool found_reshape(model m, int l, int nl)         {return is_reshape(m, l, nl);}
bool found_dense(model m, int l, int nl)           {return is_dense(m, l, nl) && !found_dense_hlsinf(m, l, nl) && !found_dense_relu(m, l, nl) && !found_dense_div_clamp_relu(m, l, nl) && !found_dense_div_clamp(m, l, nl);}
bool found_resize(model m, int l, int nl)          {return is_resize(m, l, nl);}
bool found_softmax(model m, int l, int nl)         {return is_softmax(m, l, nl);}
bool found_sigmoid(model m, int l, int nl)         {return is_sigmoid(m, l, nl);}
bool found_permute(model m, int l, int nl)         {return is_permute(m, l, nl);}
bool found_clamp(model m, int l, int nl)           {return is_clamp(m, l, nl);}
bool found_mult(model m, int l, int nl)            {return is_mult(m, l, nl);}
bool found_div(model m, int l, int nl)             {return is_div(m, l, nl);}
bool found_softplus(model m, int l, int nl)        {return is_softplus(m, l, nl);}
bool found_tanh(model m, int l, int nl)            {return is_tanh(m, l, nl);}
bool found_expand(model m, int l, int nl)          {return is_expand(m, l, nl);}
bool found_select(model m, int l, int nl)          {return is_select(m, l, nl);}
bool found_diff(model m, int l, int nl)            {return is_diff(m, l, nl);}
bool found_exp(model m, int l, int nl)             {return is_exp(m, l, nl);}
bool found_constoft(model m, int l, int nl)        {return is_constoft(m, l, nl);}
bool found_dropout(model m, int l, int nl)         {return is_dropout(m, l, nl);}
bool found_leakyrelu(model m, int l, int nl)       {return is_leakyrelu(m, l, nl);}

// Layers fussed supported on HLSinf

#ifdef ENABLE_CLAMP
bool found_conv_mult_clamp_relu_maxp(model  m, int l, int nl) {return hlsinf_shift_support && is_conv_fpga(m, l, nl) && is_mult(m, l+1, nl) && is_clamp(m, l+2, nl) && is_relu(m, l+3, nl) && is_maxp(m, l+4, nl);}
bool found_conv_mult_clamp_relu(model  m, int l, int nl) {return hlsinf_shift_support && is_conv_fpga(m, l, nl) && is_mult(m, l+1, nl) && is_clamp(m, l+2, nl) && is_relu(m, l+3, nl) && !found_conv_mult_clamp_relu_maxp(m, l, nl);}
bool found_conv_div_clamp_relu_maxp(model  m, int l, int nl) {return hlsinf_shift_support && is_conv_fpga(m, l, nl) && is_div(m, l+1, nl) && is_clamp(m, l+2, nl) && is_relu(m, l+3, nl) && is_maxp(m, l+4, nl);}
bool found_conv_div_clamp_relu(model  m, int l, int nl) {return hlsinf_shift_support && is_conv_fpga(m, l, nl) && is_div(m, l+1, nl) && is_clamp(m, l+2, nl) && is_relu(m, l+3, nl) && !found_conv_div_clamp_relu_maxp(m, l, nl);}
#else
bool found_conv_mult_clamp_relu_maxp(model  m, int l, int nl) {return false;}
bool found_conv_mult_clamp_relu(model  m, int l, int nl) {return false;}
bool found_conv_div_clamp_relu_maxp(model  m, int l, int nl) {return false;}
bool found_conv_div_clamp_relu(model  m, int l, int nl) {return false;}
#endif

bool found_conv_mult(model m, int l, int nl) {return hlsinf_shift_support && is_conv_fpga(m, l, nl) && is_mult(m, l+1, nl) && !found_conv_mult_clamp_relu(m, l, nl) && !found_conv_mult_clamp_relu_maxp(m, l, nl);}
bool found_conv_div(model m, int l, int nl) {return hlsinf_shift_support && is_conv_fpga(m, l, nl) && is_div(m, l+1, nl) && !found_conv_div_clamp_relu(m, l, nl) && !found_conv_div_clamp_relu_maxp(m, l, nl);}

bool found_conv_softplus_tanh_mult_add (model m, int l, int nl) {return is_conv_fpga(m, l, nl) && is_softplus(m, l+1, nl) && is_tanh(m, l+2, nl) && is_mult(m, l+3, nl) && is_add(m, l+4, nl);}
bool found_conv_softplus_tanh_mult     (model m, int l, int nl) {return is_conv_fpga(m, l, nl) && is_softplus(m, l+1, nl) && is_tanh(m, l+2, nl) && is_mult(m, l+3, nl) && !found_conv_softplus_tanh_mult_add(m, l, nl);}

bool found_conv_relu_bn_add_upsampling(model m, int l, int nl) {return is_conv_fpga(m, l, nl) && is_relu(m, l+1, nl) && is_bn(m, l+2, nl) && is_add(m, l+3, nl) && is_upsampling(m, l+4, nl);}
bool found_conv_relu_bn_add           (model m, int l, int nl) {return is_conv_fpga(m, l, nl) && is_relu(m, l+1, nl) && is_bn(m, l+2, nl) && is_add(m, l+3, nl) && !found_conv_relu_bn_add_upsampling(m, l, nl);}
bool found_conv_relu_bn               (model m, int l, int nl) {return is_conv_fpga(m, l, nl) && is_relu(m, l+1, nl) && is_bn(m, l+2, nl) && !found_conv_relu_bn_add(m, l, nl) && !found_conv_relu_bn_add_upsampling(m, l, nl);}
#ifdef ENABLE_UPSIZE_SUPPORT
bool found_conv_relu_maxp_resize      (model m, int l, int nl) {return is_conv_fpga(m, l, nl) && is_relu(m, l+1, nl) && is_maxp(m, l+2, nl) && is_resize(m, l+3, nl);}
#else
bool found_conv_relu_maxp_resize      (model m, int l, int nl) {return false;}
#endif

bool found_conv_relu_maxp_bn          (model m, int l, int nl) {return is_conv_fpga(m, l, nl) && is_relu(m, l+1, nl) && is_maxp(m, l+2, nl) && is_bn(m, l+3, nl);}
bool found_conv_relu_maxp             (model m, int l, int nl) {return is_conv_fpga(m, l, nl) && is_relu(m, l+1, nl) && is_maxp(m, l+2, nl) && !found_conv_relu_maxp_resize(m, l, nl) && !found_conv_relu_maxp_bn(m, l, nl);}

#ifdef ENABLE_UPSIZE_SUPPORT
bool found_conv_relu_resize           (model m, int l, int nl) {return is_conv_fpga(m, l, nl) && is_relu(m, l+1, nl) && is_resize(m, l+2, nl);}
#else
bool found_conv_relu_resize           (model m, int l, int nl) {return false;}
#endif

bool found_conv_relu                  (model m, int l, int nl) {return is_conv_fpga(m, l, nl) && is_relu(m, l+1, nl) && 
                                                                       !found_conv_relu_maxp(m, l, nl) && !found_conv_relu_maxp_resize(m, l, nl) && 
                                                                       !found_conv_relu_bn(m, l, nl) && !found_conv_relu_bn_add(m, l, nl) && !found_conv_relu_resize(m, l, nl) && !found_conv_relu_maxp_bn(m, l, nl) &&
                                                                       !found_conv_relu_bn_add_upsampling(m, l, nl);}

bool found_conv_leakyrelu             (model m, int l, int nl) {return is_conv_fpga(m, l, nl) && is_leakyrelu(m, l+1, nl);}
bool found_conv_maxp                  (model m, int l, int nl) {return is_conv_fpga(m, l, nl) && is_maxp(m, l+1, nl);}
bool found_conv_add                   (model m, int l, int nl) {return is_conv_fpga(m, l, nl) && is_add(m, l+1, nl);}
bool found_conv_bn                    (model m, int l, int nl) {return is_conv_fpga(m, l, nl) && is_bn(m, l+1, nl);}

bool found_conv                       (model m, int l, int nl) {return is_conv_fpga(m, l, nl) && !found_conv_mult_clamp_relu_maxp(m, l, nl) && !found_conv_mult_clamp_relu(m, l, nl) && !found_conv_div_clamp_relu_maxp(m, l, nl) &&
                                                                                                 !found_conv_mult(m, l, nl) && !found_conv_div(m, l, nl) &&
                                                                                                 !found_conv_div_clamp_relu(m, l, nl) && !found_conv_softplus_tanh_mult_add(m, l, nl) && !found_conv_softplus_tanh_mult(m, l, nl) && !found_conv_relu_bn_add(m, l, nl) &&
                                                                                                 !found_conv_relu_bn_add_upsampling(m, l, nl) &&
                                                                                                 !found_conv_relu_bn(m, l, nl) && !found_conv_relu_maxp(m, l, nl) && !found_conv_relu(m, l, nl) && !found_conv_leakyrelu(m, l, nl) && !found_conv_maxp(m, l, nl) &&
                                                                                                 !found_conv_add(m, l, nl) && !found_conv_bn(m, l, nl) && !found_conv_relu_maxp_resize(m, l, nl) && !found_conv_relu_resize(m, l, nl) && !found_conv_relu_maxp_bn(m, l, nl);}

bool found_pad_conv_sigmoid_tanh_maxp_add(model m, int l, int nl) {return is_pad(m, l, nl) && is_conv_fpga(m, l+1, nl) && is_sigmoid(m, l+2, nl) && is_tanh(m, l+3, nl) && is_maxp(m, l+4, nl) && is_add(m, l+5, nl);}
bool found_pad_conv_sigmoid_tanh_maxp(model m, int l, int nl) {return is_pad(m, l, nl) && is_conv_fpga(m, l+1, nl) && is_sigmoid(m, l+2, nl) && is_tanh(m, l+3, nl) && is_maxp(m, l+4, nl) && !found_pad_conv_sigmoid_tanh_maxp_add(m, l, nl);}
bool found_pad_conv_relu_maxp (model m, int l, int nl) {return is_pad(m, l, nl) && is_conv_fpga(m, l+1, nl) && is_relu(m, l+2, nl) && is_maxp(m, l+3, nl);}
bool found_pad_conv_relu      (model m, int l, int nl) {return is_pad(m, l, nl) && is_conv_fpga(m, l+1, nl) && is_relu(m, l+2, nl) && !found_pad_conv_relu_maxp(m, l, nl);}
bool found_pad_conv_maxp      (model m, int l, int nl) {return is_pad(m, l, nl) && is_conv_fpga(m, l+1, nl) && is_maxp(m, l+2, nl);}
bool found_pad_conv_leakyrelu (model m, int l, int nl) {return is_pad(m, l, nl) && is_conv_fpga(m, l+1, nl) && is_leakyrelu(m, l+2, nl);}
bool found_pad_conv_bn        (model m, int l, int nl) {return is_pad(m, l, nl) && is_conv_fpga(m, l+1, nl) && is_bn(m, l+2, nl);}
bool found_pad_conv           (model m, int l, int nl) {return is_pad(m, l, nl) && is_conv_fpga(m, l+1, nl) && !found_pad_conv_bn(m, l, nl) && !found_pad_conv_maxp(m, l, nl) && !found_pad_conv_relu(m, l, nl) && !found_pad_conv_leakyrelu(m, l, nl) && !found_pad_conv_relu_maxp(m, l, nl) && !found_pad_conv_sigmoid_tanh_maxp(m, l, nl) && !found_pad_conv_sigmoid_tanh_maxp_add(m, l, nl);}

// non-fpga layer
bool found_pad(model m, int l, int nl)             {return is_pad(m, l, nl) && !found_pad_conv(m, l, nl) && !found_pad_conv_bn(m, l, nl) && !found_pad_conv_leakyrelu(m, l, nl) && !found_pad_conv_maxp(m, l, nl) && !found_pad_conv_relu(m, l, nl) && !found_pad_conv_relu_maxp(m, l, nl) && 
                                                                               !found_pad_conv_sigmoid_tanh_maxp(m, l, nl) && !found_pad_conv_sigmoid_tanh_maxp_add(m, l, nl);}

bool found_fpga_layer(model m, int l, int nl) { return found_conv_mult_clamp_relu_maxp(m, l, nl) || found_conv_mult_clamp_relu(m, l, nl) || found_conv_div_clamp_relu_maxp(m, l, nl) || found_conv_div_clamp_relu(m, l, nl) ||
                                                         found_conv_mult(m, l, nl) || found_conv_div(m, l, nl) ||
                                                         found_conv_softplus_tanh_mult_add(m, l, nl) || found_conv_softplus_tanh_mult(m, l, nl) || 
                                                found_conv_relu_bn_add_upsampling(m, l, nl) || found_conv_relu_bn_add(m, l, nl) || found_conv_relu_bn(m, l, nl) || found_conv_relu_maxp(m, l, nl) || found_conv_relu(m, l, nl) || found_conv_leakyrelu(m, l, nl) || found_conv_maxp(m, l, nl) || found_conv_add(m, l, nl) || found_conv_bn(m, l, nl) || found_conv(m, l, nl) || 
                                                found_conv_relu_maxp_resize(m, l, nl) || found_conv_relu_resize(m, l, nl) || found_conv_relu_maxp_bn(m, l, nl) ||
                                                found_pad_conv_sigmoid_tanh_maxp_add(m, l, nl) || found_pad_conv_sigmoid_tanh_maxp(m, l, nl) || found_pad_conv_relu_maxp(m, l, nl) || found_pad_conv_relu(m, l, nl) || found_pad_conv_maxp(m, l, nl) || found_pad_conv_leakyrelu(m, l, nl) || 
                                                found_pad_conv_bn(m, l, nl) || found_pad_conv(m, l, nl) || found_dense_hlsinf(m, l, nl) || found_dense_relu(m, l, nl) || found_dense_div_clamp(m, l, nl) || found_dense_div_clamp_relu(m, l, nl);}
bool found_fpga_dense_layer(model m, int l, int nl) {return found_dense_hlsinf(m, l, nl) || found_dense_relu(m, l, nl) || found_dense_div_clamp(m, l, nl) || found_dense_div_clamp_relu(m, l, nl);}                                                
bool found_with_add(model m, int l, int nl) { return found_conv_relu_bn_add_upsampling(m, l, nl) || found_conv_relu_bn_add(m, l, nl) || found_conv_softplus_tanh_mult_add(m, l, nl) || found_conv_add(m, l, nl);}

LDense *get_dense_layer(model m, int l, int nl) {
  if (found_dense_relu(m, l, nl) || found_dense_hlsinf(m, l, nl) || found_dense_div_clamp(m, l, nl) || found_dense_div_clamp_relu(m, l, nl)) return (LDense *)m->layers[l];
  
  return (LDense *)NULL;
}

LConv *get_conv_layer(model m, int l, int nl) {
  if (found_conv_mult_clamp_relu_maxp(m, l, nl) || found_conv_mult_clamp_relu(m, l, nl) || found_conv_div_clamp_relu_maxp(m, l, nl) || found_conv_div_clamp_relu(m, l, nl) || 
      found_conv_mult(m, l, nl) || found_conv_div(m, l, nl) ||
      found_conv_softplus_tanh_mult_add(m, l, nl) ||
      found_conv_softplus_tanh_mult(m, l, nl) || found_conv_relu_bn_add_upsampling(m, l, nl) || found_conv_relu_bn_add(m, l, nl) || found_conv_relu_bn(m, l, nl) || found_conv_relu_maxp(m, l, nl) || found_conv_relu(m, l, nl) || found_conv_leakyrelu(m, l, nl) ||
      found_conv_maxp(m, l, nl) || found_conv_add(m, l, nl) || found_conv_bn(m, l, nl) || found_conv(m, l, nl) || found_conv_relu_maxp_resize(m, l, nl) || found_conv_relu_resize(m, l, nl) || found_conv_relu_maxp_bn(m, l, nl)) return (LConv *)m->layers[l];

  if (found_pad_conv_sigmoid_tanh_maxp_add(m, l, nl) || found_pad_conv_sigmoid_tanh_maxp(m, l, nl) || found_pad_conv_relu_maxp(m, l, nl) || found_pad_conv_relu(m, l, nl) || found_pad_conv_maxp(m, l, nl) ||
      found_pad_conv_leakyrelu(m, l, nl) || found_pad_conv(m, l, nl) || found_pad_conv_bn(m, l, nl)) return (LConv *)m->layers[l+1];

  return (LConv *)NULL;
}

LPad *get_pad_layer(model m, int l, int nl) {
 if (found_pad_conv_sigmoid_tanh_maxp_add(m, l, nl) || found_pad_conv_sigmoid_tanh_maxp(m, l, nl) || found_pad_conv_relu_maxp(m, l, nl) || found_pad_conv_relu(m, l, nl) || found_pad_conv_maxp(m, l, nl) ||
      found_pad_conv_leakyrelu(m, l, nl) || found_pad_conv(m, l, nl) || found_pad_conv_bn(m, l, nl)) return (LPad *)m->layers[l];

  return (LPad *)NULL;
}

LClamp *get_clamp_layer(model m, int l, int nl) {
  if (found_conv_mult_clamp_relu_maxp(m, l, nl)) return (LClamp *)m->layers[l+2];
  if (found_conv_mult_clamp_relu(m, l, nl)) return (LClamp *)m->layers[l+2];
  if (found_conv_div_clamp_relu_maxp(m, l, nl)) return (LClamp *)m->layers[l+2];
  if (found_conv_div_clamp_relu(m, l, nl)) return (LClamp *)m->layers[l+2];
  if (found_dense_div_clamp(m, l, nl)) return (LClamp *)m->layers[l+2];
  if (found_dense_div_clamp_relu(m, l, nl)) return (LClamp *)m->layers[l+2];
  return (LClamp *)NULL;
}

LActivation *get_leakyrelu_layer(model m, int l, int nl) {
  if (found_conv_leakyrelu(m, l, nl)) return (LActivation *)m->layers[l+1];
  if (found_pad_conv_leakyrelu(m, l, nl)) return (LActivation *)m->layers[l+1];
  return (LActivation *)NULL;
}

LMult *get_mult_layer(model m, int l, int nl) {
  if (found_conv_mult_clamp_relu_maxp(m, l, nl)) return (LMult *)m->layers[l+1];
  if (found_conv_mult_clamp_relu(m, l, nl)) return (LMult *)m->layers[l+1];
  if (found_conv_mult(m, l, nl)) return (LMult *)m->layers[l+1];
  if (found_conv_div_clamp_relu_maxp(m, l, nl)) return (LMult *)m->layers[l+1];
  if (found_conv_div_clamp_relu(m, l, nl)) return (LMult *)m->layers[l+1];
  if (found_conv_div(m, l, nl)) return (LMult *)m->layers[l+1];
  if (found_dense_div_clamp(m, l, nl)) return (LMult *)m->layers[l+1];
  if (found_dense_div_clamp_relu(m, l, nl)) return (LMult *)m->layers[l+1];
  return (LMult *)NULL;
}

LAdd *get_add_layer(model m, int l, int nl) {
  if (found_conv_relu_bn_add(m, l, nl)) return (LAdd *)m->layers[l+3];
  if (found_conv_relu_bn_add_upsampling(m, l, nl)) return (LAdd *)m->layers[l+3];
  if (found_conv_add(m, l, nl)) return (LAdd *)m->layers[l+1];
  if (found_conv_softplus_tanh_mult_add(m, l, nl)) return (LAdd *)m->layers[l+4];
  return NULL;
}

Layer *get_prev_layer_to_add_layer(model m, int l, int nl) {
  if (found_conv_relu_bn_add(m, l, nl)) return (Layer *)m->layers[l+2];
  if (found_conv_relu_bn_add_upsampling(m, l, nl)) return (Layer *)m->layers[l+2];
  if (found_conv_add(m, l, nl)) return (Layer *)m->layers[l];
  if (found_conv_softplus_tanh_mult_add(m, l, nl)) return (LAdd *)m->layers[l+3];
  return NULL;
}

LBatchNorm *get_bn_layer(model m, int l, int nl) {
  if (found_conv_relu_bn(m, l, nl)) return (LBatchNorm *)m->layers[l+2];
  if (found_conv_relu_bn_add(m, l, nl)) return (LBatchNorm *)m->layers[l+2];
  if (found_conv_relu_bn_add_upsampling(m, l, nl)) return (LBatchNorm *)m->layers[l+2];
  if (found_conv_bn(m, l, nl)) return (LBatchNorm *)m->layers[l+1];
  if (found_pad_conv_bn(m, l, nl)) return (LBatchNorm *)m->layers[l+2];
  if (found_conv_relu_maxp_bn(m, l, nl)) return (LBatchNorm *)m->layers[l+3];
  return NULL;
}

int get_h(model m, int l, int nl) {
  LPad *pad = get_pad_layer(m, l, nl);
  LConv *conv = get_conv_layer(m, l, nl);
  LDense *dense = get_dense_layer(m, l, nl);
  if ((pad == NULL) && (conv != NULL)) return conv->cd->I->shape[2]; 
  if ((pad != NULL) && (conv != NULL)) return conv->cd->I->shape[2]-pad->padding[0]-pad->padding[2];
  if (dense != NULL) return 3;
  return -1;
}
int get_w(model m, int l, int nl) {
  LPad *pad = get_pad_layer(m, l, nl);
  LConv *conv = get_conv_layer(m, l, nl); 
  LDense *dense = get_dense_layer(m, l, nl);
  if ((pad == NULL) && (conv != NULL)) return conv->cd->I->shape[3]; 
  if ((pad != NULL) && (conv != NULL)) return conv->cd->I->shape[3]-pad->padding[1]-pad->padding[3];
  if (dense != NULL) return 3;
  return -1;
}
int get_i(model m, int l, int nl) {
  LConv *conv = get_conv_layer(m, l, nl); 
  LDense *dense = get_dense_layer(m, l, nl);
  if (conv != NULL) return conv->cd->I->shape[1]; 
  if (dense != NULL) return (dense->input->shape[1] + 8) / 9;
  return -1;
}
int get_o(model m, int l, int nl) {
  LConv *conv = get_conv_layer(m, l, nl); 
  LDense *dense = get_dense_layer(m, l, nl);
  if (conv != NULL) return conv->cd->O->shape[1]; 
  if (dense != NULL) return dense->W->shape[1];
  return -1;
}
int get_kh(model m, int l, int nl) {
  LConv *conv = get_conv_layer(m, l, nl); 
  LDense *dense = get_dense_layer(m, l, nl);
  if (conv != NULL) return conv->cd->kr; 
  if (dense != NULL) return 3;
  return -1;
}
int get_kw(model m, int l, int nl) {
  LConv *conv = get_conv_layer(m, l, nl); 
  LDense *dense = get_dense_layer(m, l, nl);
  if (conv != NULL) return conv->cd->kc; 
  if (dense != NULL) return 3;
  return -1;
}
int get_sh(model m, int l, int nl) {LConv *conv = get_conv_layer(m, l, nl); if (conv != NULL) return conv->cd->sr; return 1;}
int get_sw(model m, int l, int nl) {LConv *conv = get_conv_layer(m, l, nl); if (conv != NULL) return conv->cd->sc; return 1;}

int get_pt(model m, int l, int nl) {
  LPad *pad = get_pad_layer(m, l, nl); 
  LConv *conv = get_conv_layer(m, l, nl); 
  if ((pad != NULL) && (conv == NULL)) return pad->padding[0];
  if ((pad == NULL) && (conv != NULL)) return conv->cd->padrt;
  if ((pad != NULL) && (conv != NULL)) return pad->padding[0] + conv->cd->padrt;
  return 0;
}

int get_pb(model m, int l, int nl) {
  LPad *pad = get_pad_layer(m, l, nl); 
  LConv *conv = get_conv_layer(m, l, nl); 
  if ((pad != NULL) && (conv == NULL)) return pad->padding[2];
  if ((pad == NULL) && (conv != NULL)) return conv->cd->padrb;
  if ((pad != NULL) && (conv != NULL)) return pad->padding[2] + conv->cd->padrb;
  return 0;
}

int get_pl(model m, int l, int nl) {
  LPad *pad = get_pad_layer(m, l, nl); 
  LConv *conv = get_conv_layer(m, l, nl); 
  if ((pad != NULL) && (conv == NULL)) return pad->padding[3];
  if ((pad == NULL) && (conv != NULL)) return conv->cd->padcl;
  if ((pad != NULL) && (conv != NULL)) return pad->padding[3] + conv->cd->padcl;
  return 0;
}

int get_pr(model m, int l, int nl) {
  LPad *pad = get_pad_layer(m, l, nl); 
  LConv *conv = get_conv_layer(m, l, nl); 
  if ((pad != NULL) && (conv == NULL)) return pad->padding[1];
  if ((pad == NULL) && (conv != NULL)) return conv->cd->padcr;
  if ((pad != NULL) && (conv != NULL)) return pad->padding[1] + conv->cd->padcr;
  return 0;
}

int get_enable_relu(model m, int l, int nl) {
  return found_conv_mult_clamp_relu_maxp(m, l, nl) || found_conv_mult_clamp_relu(m, l, nl) || found_conv_div_clamp_relu_maxp(m, l, nl) || found_conv_div_clamp_relu(m, l, nl) || found_conv_relu_bn_add(m, l, nl) || found_conv_relu_bn_add_upsampling(m, l, nl) ||
         found_conv_relu_bn(m, l, nl) || found_conv_relu_maxp(m, l, nl) || found_conv_relu(m, l, nl) || found_conv_leakyrelu(m, l, nl) || found_pad_conv_relu_maxp(m, l, nl) || found_pad_conv_relu(m, l, nl) || found_pad_conv_leakyrelu(m, l, nl) ||
         found_conv_relu_maxp_resize(m, l, nl) || found_conv_relu_resize(m, l, nl) || found_dense_relu(m, l, nl) || found_dense_div_clamp_relu(m, l, nl) || found_conv_relu_maxp_bn(m, l, nl);
}
int get_enable_maxp(model m, int l, int nl) {return found_conv_maxp(m, l, nl) || found_conv_relu_maxp(m, l, nl) || found_conv_mult_clamp_relu_maxp(m, l, nl) || found_conv_div_clamp_relu_maxp(m, l, nl) || found_conv_relu_maxp_resize(m, l, nl) || found_conv_relu_maxp_bn(m, l, nl);}
int get_enable_avgp(model m, int l, int nl) {return false;}
int get_enable_clipping(model m, int l, int nl) {return found_conv_mult_clamp_relu_maxp(m, l, nl) || found_conv_mult_clamp_relu(m, l, nl) || found_conv_div_clamp_relu_maxp(m, l, nl) || found_conv_div_clamp_relu(m, l, nl) || found_dense_div_clamp(m, l, nl) || found_dense_div_clamp_relu(m, l, nl);}
int get_min_clip(model m, int l, int nl) {
  LClamp *clamp = get_clamp_layer(m, l, nl);
  if (clamp != NULL) return clamp->min;
  return 0;
}
int get_max_clip(model m, int l, int nl) {
  LClamp *clamp = get_clamp_layer(m, l, nl);
  if (clamp != NULL) return clamp->max;
  return 0;
}
  
int get_enable_shift(model m, int l, int nl) {return found_conv_mult_clamp_relu_maxp(m, l, nl) || found_conv_mult_clamp_relu(m, l, nl) || 
                                                     found_conv_mult(m, l, nl) || found_conv_div(m, l, nl) ||
                                                     found_conv_div_clamp_relu_maxp(m, l, nl) || found_conv_div_clamp_relu(m, l, nl) ||
                                                     found_dense_div_clamp(m, l, nl) || found_dense_div_clamp_relu(m, l, nl);}
int get_pos_shift(model m, int l, int nl) {
  LMult *mult = get_mult_layer(m, l, nl);
  if (mult != NULL) {
    int pos;
    if (mult->val >= 1.0f) pos = abs(log2(1 / mult->val)); else pos = abs(log2(1 / mult->val));
    #ifdef FPGA_DEBUG
    printf("shift: value %f pos %d\n", mult->val, pos);
    #endif
    return pos;
  }
  return 0;
}
int get_dir_shift(model m, int l, int nl) {
  LMult *mult = get_mult_layer(m, l, nl);
  if (mult != NULL) {
    int dir;
    if (mult->val < 1.0f) dir = 0; /* left */ else dir = 1; /* right */
    #ifdef FPGA_DEBUG
    printf("shift: value %f dir %d\n", mult->val, dir);
    #endif
    return dir;
  }
  return -1;
}

int get_enable_add(model m, int l, int nl) {return found_conv_softplus_tanh_mult_add(m, l, nl) || found_conv_relu_bn_add_upsampling(m, l, nl) || found_conv_relu_bn_add(m, l, nl) || found_conv_add(m, l, nl) || found_pad_conv_sigmoid_tanh_maxp_add(m, l, nl);}
int get_enable_stm(model m, int l, int nl) {return found_conv_softplus_tanh_mult_add(m, l, nl) ||  found_conv_softplus_tanh_mult(m, l, nl); /* ||  found_pad_conv_sigmoid_tanh_maxp_add(m, l, nl) ||  found_pad_conv_sigmoid_tanh_maxp(m, l, nl);*/}
int get_enable_bn(model m, int l, int nl) {return found_conv_bn(m, l, nl) || found_conv_relu_bn(m, l, nl) || found_conv_relu_bn_add_upsampling(m, l, nl) || found_conv_relu_bn_add(m, l, nl) || found_pad_conv_bn(m, l, nl) || found_conv_relu_maxp_bn(m, l, nl);}

float get_relu_factor(model m, int l, int nl) {
  LActivation *act = get_leakyrelu_layer(m, l, nl);
  if (act != NULL) return act->params[0];
  return 0;
}

int get_upscale(model m, int l, int nl) {return found_conv_relu_resize(m, l, nl) || found_conv_relu_maxp_resize(m, l, nl) || found_conv_relu_bn_add_upsampling(m, l, nl);}

void get_name(model m, int l, int nl, char *str) { // TODO
  if (found_conv_relu(m, l, nl))                        strcpy(str, "HLSinf (Conv + ReLu)");
  else if (found_conv(m, l, nl))                        strcpy(str, "HLSinf (Conv)");
  else if (found_conv_relu_bn_add_upsampling(m, l, nl)) strcpy(str, "HLSinf (Conv + ReLu + BatchNorm + Add + Upsampling)");
  else if (found_conv_relu_bn_add(m, l, nl))            strcpy(str, "HLSinf (Conv + ReLu + BatchNorm + Add)");
  else if (found_conv_relu_bn(m, l, nl))                strcpy(str, "HLSinf (Conv + ReLu + BatchNorm)");
  else if (found_conv_relu_maxp(m, l, nl))              strcpy(str, "HLSinf (Conv + ReLu + MaxPool)");
  else if (found_pad_conv(m, l, nl))                    strcpy(str, "HLSinf (Padding + Conv)");
  else if (found_pad_conv_relu(m, l, nl))               strcpy(str, "HLSinf (Conv + ReLu)");
  else if (found_pad_conv_leakyrelu(m, l, nl))          strcpy(str, "HLSinf (Padding + Conv + LeakyReLu)");
  else if (found_pad_conv_bn(m, l, nl))                 strcpy(str, "HLSinf (Padding + Conv + BatchNorm)");
  else if (found_conv_add(m, l, nl))                    strcpy(str, "HLSinf (Conv + Add)");
  else if (found_conv_mult_clamp_relu(m, l, nl))        strcpy(str, "HLSinf (Conv + Mult + Clamp + ReLu)");
  else if (found_conv_mult_clamp_relu_maxp(m, l, nl))   strcpy(str, "HLSinf (Conv + Mult + Clamp + ReLu + MaxPool)");
  else if (found_conv_mult(m, l, nl))                   strcpy(str, "HLSinf (Conv + Mult)");
  else if (found_conv_div(m, l, nl))                    strcpy(str, "HLSinf (Conv + Div)");
  else if (found_conv_div_clamp_relu(m, l, nl))         strcpy(str, "HLSinf (Conv + Div + Clamp + ReLu)");
  else if (found_conv_div_clamp_relu_maxp(m, l, nl))    strcpy(str, "HLSinf (Conv + Div + Clamp + ReLu + MaxPool)");
  else if (found_conv_relu_maxp_resize(m, l, nl))       strcpy(str, "HLSinf (Conv + ReLu + MaxPool + Resize");
  else if (found_conv_relu_maxp_bn(m, l, nl))           strcpy(str, "HLSinf (Conv + ReLu + MaxPool + BatchNorm)");
  else if (found_conv_relu_resize(m, l, nl))            strcpy(str, "HLSinf (Conv + ReLu + Resize)");
  else if (found_dense_hlsinf(m, l, nl))                strcpy(str, "HLSinf (Dense)");
  else if (found_dense_relu(m, l, nl))                  strcpy(str, "HLSinf (Dense + ReLu");
  else if (found_conv_softplus_tanh_mult(m, l, nl))     strcpy(str, "HLSinf (Conv + Softplus + Tanh + Mult)");
  else if (found_conv_softplus_tanh_mult_add(m, l, nl)) strcpy(str, "HLSinf (Conv + Softplus + Tanh + Mult + Add)");
  else if (found_conv_leakyrelu(m, l, nl))              strcpy(str, "HLSinf (Conv + LeakyReLu)");
  else if (found_dense_div_clamp(m, l, nl))             strcpy(str, "HLSinf (Dense + Div + Clamp");
  else if (found_dense_div_clamp_relu(m, l, nl))        strcpy(str, "HLSinf (Dense + Div + Clamp + ReLu");
  else if (found_conv_bn(m, l, nl))                     strcpy(str, "HLSinf (Conv + BatchNorm)");
  else                                                  strcpy(str, "?????");
}
int get_num_layers_fused(model m, int l, int nl) { // TODO
  if (found_input(m, l, nl)) return 1;
  if (found_bn(m, l, nl)) return 1;
  if (found_relu(m, l, nl)) return 1;
  if (found_conv(m, l, nl)) return 1;
  if (found_conv_cpu(m, l, nl)) return 1;
  if (found_maxp(m, l, nl)) return 1;
  if (found_avgp(m, l, nl)) return 1;
  if (found_add(m, l, nl)) return 1;
  if (found_linear(m, l, nl)) return 1;
  if (found_upsampling(m, l, nl)) return 1;
  if (found_reshape(m, l, nl)) return 1;
  if (found_dense(m, l, nl)) return 1;
  if (found_resize(m, l, nl)) return 1;
  if (found_softmax(m, l, nl)) return 1;
  if (found_sigmoid(m, l, nl)) return 1;
  if (found_permute(m, l, nl)) return 1;
  if (found_concat(m, l, nl)) return 1;
  if (found_clamp(m, l, nl)) return 1;
  if (found_mult(m, l, nl)) return 1;
  if (found_div(m, l, nl)) return 1;
  if (found_softplus(m, l, nl)) return 1;
  if (found_tanh(m, l, nl)) return 1;
  if (found_dense_hlsinf(m, l, nl)) return 1;
  if (found_expand(m, l, nl)) return 1;
  if (found_select(m, l, nl)) return 1;
  if (found_diff(m, l, nl)) return 1;
  if (found_exp(m, l, nl)) return 1;
  if (found_constoft(m, l, nl)) return 1;
  if (found_dropout(m, l, nl)) return 1;
  if (found_leakyrelu(m, l, nl)) return 1;
  if (found_pad(m, l, nl)) return 1;
  if (found_conv_relu(m, l, nl)) return 2;
  if (found_conv_add(m, l, nl)) return 2;
  if (found_dense_relu(m, l, nl)) return 2;
  if (found_conv_bn(m, l, nl)) return 2;
  if (found_conv_relu_bn(m, l, nl)) return 3;
  if (found_conv_relu_bn_add(m, l, nl)) return 4;
  if (found_conv_relu_maxp(m, l, nl)) return 3;
  if (found_dense_div_clamp(m, l, nl)) return 3;
  if (found_pad_conv(m, l, nl)) return 2;
  if (found_conv_mult(m, l, nl)) return 2;
  if (found_conv_div(m, l, nl)) return 2;
  if (found_conv_leakyrelu(m, l, nl)) return 2;
  if (found_pad_conv_relu(m, l, nl)) return 3;
  if (found_pad_conv_leakyrelu(m, l, nl)) return 3;
  if (found_pad_conv_bn(m, l, nl)) return 3;
  if (found_conv_relu_resize(m, l, nl)) return 3;
  if (found_conv_relu_maxp_resize(m, l, nl)) return 4;
  if (found_conv_relu_maxp_bn(m, l, nl)) return 4;
  if (found_conv_mult_clamp_relu(m, l, nl)) return 4;
  if (found_dense_div_clamp_relu(m, l, nl)) return 4;
  if (found_conv_mult_clamp_relu_maxp(m, l, nl)) return 5;
  if (found_conv_div_clamp_relu(m, l, nl)) return 4;
  if (found_conv_div_clamp_relu_maxp(m, l, nl)) return 5;
  if (found_conv_softplus_tanh_mult(m, l, nl)) return 4;
  if (found_conv_softplus_tanh_mult_add(m, l, nl)) return 5;
  if (found_conv_relu_bn_add_upsampling(m, l, nl)) return 5;
  printf("error, num layers does not recognize how many\n");
  exit(1);
  return 0;
}

// model for fpga
model toFPGA(model m_src, int kernel_version, int kernel_subversion) {
    #ifdef cFPGA

    int dummy;
    int dummy1;
    vlayer first;       // first layer
    vlayer last;        // last layer
    layer prev_layer;  // for network building process (previous layer)

    current_associated_layers = 0;
    for (int x = 0; x < MAX_ASSOCIATED_LAYERS; x++) {
      associated_layers[x].src = NULL; associated_layers[x].dst_ghwc = NULL; associated_layers[x].dst_chw = NULL; associated_layers[x].layer_id_ghwc = -1; associated_layers[x].layer_id_chw = -1;
      associated_source_layer[x].src = NULL; associated_source_layer[x].conv = NULL; associated_source_layer[x].bn = NULL; associated_source_layer[x].dense = NULL; associated_source_layer[x].dst = NULL;
    }

    fpga_init(kernel_version, kernel_subversion);

      // constants
      const int CPI = hlsinf_cpi;
      const int CPO = hlsinf_cpo;

      // New model and number of layers
      Net *net = new Net();
      int num_layers = m_src->layers.size();

      // we list the whole source model
#ifdef FPGA_DEBUG
      printf("-----------------------------------\n");
      printf("Layers (name, address, and its parents):\n");
      int l=0;
      while (l < num_layers) {
        Layer *cl;
        cl = m_src->layers[l];
        cout << "Layer " << l << " name: " << cl->name << " address: " << cl << " parents: ";
        for(int p = 0; p < cl->parent.size();p++){
          cout << cl->parent[p] << " ";
        }
        cout << "\n";
        l++;
      }
      printf("-----------------------------------\n");
      printf("Input Layers \n");
      printf("%p\n", m_src);
      printf("Output Layers (%d layers)\n", m_src->lout.size());
      for(int lout = 0; lout < m_src->lout.size(); lout++)
        cout << m_src->lout[lout]->name << "\n";
#endif
  // we sweep all the model in search of layers that can be merged
  int l_src = 0;
  int l_dst = 0;

  while (l_src<num_layers) {

    #ifdef FPGA_DEBUG
    printf("inspecting from layer %d\n", l_src);
    #endif

    Layer *cl      = m_src->layers[l_src];
    Layer *nl      = (l_src < num_layers-1)   ? m_src->layers[l_src+1] : NULL;
    Layer *nnl     = (l_src+1 < num_layers-1) ? m_src->layers[l_src+2] : NULL; 
    Layer *nnnl    = (l_src+2 < num_layers-1) ? m_src->layers[l_src+3] : NULL; 
    Layer *nnnnl   = (l_src+3 < num_layers-1) ? m_src->layers[l_src+4] : NULL; 
    Layer *nnnnnl  = (l_src+4 < num_layers-1) ? m_src->layers[l_src+5] : NULL; 
    Layer *nnnnnnl = (l_src+5 < num_layers-1) ? m_src->layers[l_src+6] : NULL; 

    LConv *conv_layer = get_conv_layer(m_src, l_src, num_layers);
    LDense *dense_layer = get_dense_layer(m_src, l_src, num_layers);
    LAdd  *add_layer  = get_add_layer(m_src, l_src, num_layers);
    Layer *prev_layer_to_add_layer = get_prev_layer_to_add_layer(m_src, l_src, num_layers);

      if (found_fpga_layer(m_src, l_src, num_layers)) {
        // all these layers need a transform layer at the input if the previous layer runs on CPU
        for (int x=0; x<cl->parent.size(); x++) {
          Layer *parent_layer = fn_get_associated_layer(cl->parent[x], 1, &dummy);
          if (parent_layer == NULL) {
            // we add a transform layer
            parent_layer = fn_get_associated_layer(cl->parent[x], 0, &dummy);
            #ifdef FPGA_DEBUG
            printf("%3d: TRANSFORM  : prev %d\n", l_dst, dummy);
            #endif
            int copy_cpu_to_fpga = 1;
            int copy_fpga_to_cpu = 0;
            int transform;
            if (found_fpga_dense_layer(m_src, l_src, num_layers)) transform = 0; else transform = 1;
            Layer *new_parent_layer = Transform(parent_layer, copy_cpu_to_fpga, copy_fpga_to_cpu, transform, 1);
            fn_set_associated_layer(cl->parent[x], new_parent_layer, 1, ghwcpi_format, cpu_device, l_dst);
            l_dst++;
          }
        }
      
        if (found_with_add(m_src, l_src, num_layers)) {
          // The parents of Add layer need also to be in GHWC format
          for (int x=0; x<add_layer->parent.size(); x++) {
            if (add_layer->parent[x] != prev_layer_to_add_layer) { // we skip M layer (internal)
              Layer *parent_layer = fn_get_associated_layer(add_layer->parent[x], 1, &dummy);
              if (parent_layer == NULL) {
                // we add a transform layer
                parent_layer = fn_get_associated_layer(add_layer->parent[x], 0, &dummy);
                #ifdef FPGA_DEBUG
                printf("%3d: TRANSFORM  : prev %d\n", l_dst, dummy);
                #endif
                int copy_cpu_to_fpga = 1;
                int copy_fpga_to_cpu = 0;
                int transform = 1;
                Layer *new_parent_layer = Transform(parent_layer, copy_cpu_to_fpga, copy_fpga_to_cpu, transform, 1);
                fn_set_associated_layer(add_layer->parent[x], new_parent_layer, 1, ghwcpi_format, cpu_device, l_dst);
                l_dst++;
              }
            }
          }
        }
      } else {
        // The rest of layers need the CHW format at the inputs, therefore we check if the
        // previous layers are in CHW format and if not we add a transform layer
        //
        for (int x=0; x<cl->parent.size(); x++) {
          Layer *parent_layer = fn_get_associated_layer(cl->parent[x], 0, &dummy);
          if (parent_layer == NULL) {
            // we add a transform layer
            parent_layer = fn_get_associated_layer(cl->parent[x], 1, &dummy);
            #ifdef FPGA_DEBUG
            printf("%3d: TRANSFORM : prev %d\n", l_dst, dummy);
            #endif
            int copy_cpu_to_fpga = 0;
            int copy_fpga_to_cpu = 1;
            int transform;
            int prev_format = fn_get_associated_layer_format(cl->parent[x]);
            if (prev_format == chw_format) transform = 0; else transform = 1;
            Layer *new_parent_layer = Transform(parent_layer, copy_cpu_to_fpga, copy_fpga_to_cpu, transform, 0);
            fn_set_associated_layer(cl->parent[x], new_parent_layer, 0, chw_format, cpu_device, l_dst);
            l_dst++;
          }
        }
      }
    //}

    // build up stage, we create a merged layer out of our findings

  // parameters
  int h                 = get_h(m_src, l_src, num_layers);
  int w                 = get_w(m_src, l_src, num_layers);
  int ichannels         = get_i(m_src, l_src, num_layers);
  ichannels             = ceil((float)ichannels/CPI) * CPI; 
  int ochannels         = get_o(m_src, l_src, num_layers);
  ochannels             = ceil((float)ochannels/CPI) * CPI; 
  int kh                = get_kh(m_src, l_src, num_layers);
  int kw                = get_kw(m_src, l_src, num_layers);
  int sh                = get_sh(m_src, l_src, num_layers);
  int sw                = get_sw(m_src, l_src, num_layers);
  int pt                = get_pt(m_src, l_src, num_layers);
  int pb                = get_pb(m_src, l_src, num_layers);
  int pl                = get_pl(m_src, l_src, num_layers);
  int pr                = get_pr(m_src, l_src, num_layers);
  int enable_relu       = get_enable_relu(m_src, l_src, num_layers);
  float relu_factor     = get_relu_factor(m_src, l_src, num_layers);
  int enable_batch_norm = get_enable_bn(m_src, l_src, num_layers);
  int enable_maxp       = get_enable_maxp(m_src, l_src, num_layers);
  int enable_avgp       = get_enable_avgp(m_src, l_src, num_layers);
  int enable_clipping   = get_enable_clipping(m_src, l_src, num_layers);
  int min_clip          = get_min_clip(m_src, l_src, num_layers);
  int max_clip          = get_max_clip(m_src, l_src, num_layers);
  int enable_shift      = get_enable_shift(m_src, l_src, num_layers);
  int pos_shift         = get_pos_shift(m_src, l_src, num_layers);
  int dir_shift         = get_dir_shift(m_src, l_src, num_layers);
  int enable_add        = get_enable_add(m_src, l_src, num_layers);
  int enable_stm        = get_enable_stm(m_src, l_src, num_layers);
  int enable_upscale    = get_upscale(m_src, l_src, num_layers);
  char str_name[50];
  get_name(m_src, l_src, num_layers, str_name);
  int num_layers_fused  = get_num_layers_fused(m_src, l_src, num_layers);

  //#ifdef FPGA_DEBUG
  //printf("h %d w %d ich %d och %d kh %d kw %d sh %d sw %d pt %d pb %d pl %d pr %d relu %d relu_factor %f bn %d maxp %d avgp %d clip %d [%d, %d] shift %d pos_shift %d dir_shift %d add %d stm %d\n", h, w, ichannels, ochannels, kh, kw, sh, sw, pt, pb, pl, pr, enable_relu, relu_factor,
  //            enable_batch_norm, enable_maxp, enable_avgp, enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_add, enable_stm);
  //#endif

  if (found_conv_relu(m_src, l_src, num_layers) || found_conv_leakyrelu(m_src, l_src, num_layers) || found_conv_maxp(m_src, l_src, num_layers) ||
      found_conv_bn(m_src, l_src, num_layers) || found_conv_relu_maxp(m_src, l_src, num_layers) || found_conv_mult_clamp_relu(m_src, l_src, num_layers) ||
      found_conv_mult(m_src, l_src, num_layers) || found_conv_div(m_src, l_src, num_layers) ||
      found_conv_mult_clamp_relu_maxp(m_src, l_src, num_layers) || found_conv_div_clamp_relu(m_src, l_src, num_layers) || found_conv_div_clamp_relu_maxp(m_src, l_src, num_layers) ||
      found_conv_relu_maxp_resize(m_src, l_src, num_layers) || found_conv_relu_resize(m_src, l_src, num_layers) || found_conv_relu_maxp_bn(m_src, l_src, num_layers) ||
      found_conv_relu_bn(m_src, l_src, num_layers) || /*found_div_mult_sum_multit_sum_mult_conv(m_src, l_src, num_layers) ||*/ found_conv(m_src, l_src, num_layers) ||
      found_conv_softplus_tanh_mult(m_src, l_src, num_layers) || found_conv_relu_bn_add_upsampling(m_src, l_src, num_layers) || found_conv_relu_bn_add(m_src, l_src, num_layers) || found_conv_softplus_tanh_mult_add(m_src, l_src, num_layers) ||
      found_pad_conv(m_src, l_src, num_layers) || found_pad_conv_relu(m_src, l_src, num_layers) || found_pad_conv_leakyrelu(m_src, l_src, num_layers) ||
      found_pad_conv_maxp(m_src, l_src, num_layers) || found_pad_conv_relu_maxp(m_src, l_src, num_layers) || found_pad_conv_sigmoid_tanh_maxp(m_src, l_src, num_layers) ||
      found_pad_conv_sigmoid_tanh_maxp_add(m_src, l_src, num_layers) | found_conv(m_src, l_src, num_layers) || found_conv_add(m_src, l_src, num_layers) ||
      found_dense_hlsinf(m_src, l_src, num_layers) || found_dense_relu(m_src, l_src, num_layers) || found_dense_div_clamp(m_src, l_src, num_layers) || found_dense_div_clamp_relu(m_src, l_src, num_layers) ||
      found_pad_conv_bn(m_src, l_src, num_layers)) {

    vector<Layer *> parent;   
    if (add_layer != NULL) {
      if (add_layer->parent.size() != 2) msg("Error: LAdd layer with more than two parents is not supported in the FPGA ");
      // Parents
      Layer *first_layer = cl;
      parent.push_back(fn_get_associated_layer(first_layer->parent[0], 1, &dummy));
      if (add_layer->parent[0] != prev_layer_to_add_layer) parent.push_back(fn_get_associated_layer(add_layer->parent[0], 1, &dummy1));
      else parent.push_back(fn_get_associated_layer(add_layer->parent[1], 1, &dummy1));
    }

    Layer *fpga_parent;    // dst parent layer
    fpga_parent = fn_get_associated_layer(cl->parent[0], 1, &dummy);
    if (fpga_parent == NULL) fpga_parent = fn_get_associated_layer(cl->parent[0], 0, &dummy); // It may be a dense layer with no previous transformation needed

    #ifdef FPGA_DEBUG
    printf("%3d: %s         : prev %d\n", l_dst, str_name, dummy);
    #endif  

    int mem_level = (conv_layer != NULL) ? conv_layer->cd->mem_level : dense_layer->mem_level;
    int dense_operation;
    if (conv_layer != NULL) dense_operation = 0; else dense_operation = 1;

    if (add_layer != NULL) {
      prev_layer = new LHLSinf(parent, 
                             h, w, ichannels, ochannels, kh, kw, sh, sw, pt, pb, pl, pr, enable_relu, relu_factor,
                             enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_stm, enable_maxp, enable_avgp,
                             enable_batch_norm, enable_add, enable_upscale, dense_operation, str_name, DEV_CPU, mem_level);

    } else {
      prev_layer = new LHLSinf(fpga_parent, 
                             h, w, ichannels, ochannels, kh, kw, sh, sw, pt, pb, pl, pr, enable_relu, relu_factor,
                             enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_stm, enable_maxp, enable_avgp,
                             enable_batch_norm, enable_add, enable_upscale, dense_operation, str_name, DEV_CPU, mem_level);

    }

    int format = 1;
    int fmt;
    if (dense_layer != NULL) fmt = chw_format; else fmt = ghwcpi_format;  // dense layer implemented with HLSinf do not use GIHWCPI format

    fn_set_associated_layer(cl, prev_layer, format, fmt, fpga_device, l_dst);
    if (num_layers_fused > 1) fn_set_associated_layer(nl, prev_layer, format, fmt, fpga_device, l_dst);
    if (num_layers_fused > 2) fn_set_associated_layer(nnl, prev_layer, format, fmt, fpga_device, l_dst);
    if (num_layers_fused > 3) fn_set_associated_layer(nnnl, prev_layer, format, fmt, fpga_device, l_dst);
    if (num_layers_fused > 4) fn_set_associated_layer(nnnnl, prev_layer, format, fmt, fpga_device, l_dst);
    if (num_layers_fused > 5) fn_set_associated_layer(nnnnnl, prev_layer, format, fmt, fpga_device, l_dst);
    if (num_layers_fused > 6) fn_set_associated_layer(nnnnnnl, prev_layer, format, fmt, fpga_device, l_dst);
    associated_source_layer[l_dst].src = cl;
    associated_source_layer[l_dst].dst = prev_layer;
    associated_source_layer[l_dst].conv = get_conv_layer(m_src, l_src, num_layers);
    associated_source_layer[l_dst].bn   = get_bn_layer(m_src, l_src, num_layers);
    associated_source_layer[l_dst].dense= get_dense_layer(m_src, l_src, num_layers);
    l_dst++;
 } else if (found_input(m_src, l_src, num_layers) || found_maxp(m_src, l_src, num_layers) || found_upsampling(m_src, l_src, num_layers) || found_linear(m_src, l_src, num_layers) || 
            found_conv_cpu(m_src, l_src, num_layers) || found_relu(m_src, l_src, num_layers) || found_bn(m_src, l_src, num_layers) || found_avgp(m_src, l_src, num_layers) ||
            found_reshape(m_src, l_src, num_layers) || found_dense(m_src, l_src, num_layers) || found_resize(m_src, l_src, num_layers) || found_softmax(m_src, l_src, num_layers) || found_sigmoid(m_src, l_src, num_layers) ||
            found_permute(m_src, l_src, num_layers) || found_concat(m_src, l_src, num_layers) || found_clamp(m_src, l_src, num_layers) || found_mult(m_src, l_src, num_layers) ||
            found_div(m_src, l_src, num_layers) || found_softplus(m_src, l_src, num_layers) || found_tanh(m_src, l_src, num_layers) || found_expand(m_src, l_src, num_layers) || found_select(m_src, l_src, num_layers) || found_diff(m_src, l_src, num_layers) || found_exp(m_src, l_src, num_layers) ||
            found_constoft(m_src, l_src, num_layers) || found_dropout(m_src, l_src, num_layers) || found_leakyrelu(m_src, l_src, num_layers) || found_pad(m_src, l_src, num_layers)) {

    LConv *layer_src_conv = found_conv_cpu(m_src, l_src, num_layers) ? (LConv *)cl : NULL;
    LMaxPool *layer_src_maxp = found_maxp(m_src, l_src, num_layers) ? (LMaxPool *)cl : NULL;
    LAveragePool *layer_src_avgp = found_avgp(m_src, l_src, num_layers) ? (LAveragePool *)cl : NULL;
    LUpSampling *layer_src_upsampling = found_upsampling(m_src, l_src, num_layers) ? (LUpSampling *)cl : NULL;
    LActivation *layer_src_linear = found_linear(m_src, l_src, num_layers) ? (LActivation *)cl : NULL;
    LBatchNorm *layer_src_bn = found_bn(m_src, l_src, num_layers) ? (LBatchNorm *)cl : NULL;
    LReshape *layer_src_reshape = found_reshape(m_src, l_src, num_layers) ? (LReshape *)cl : NULL;
    LDense *layer_src_dense = found_dense(m_src, l_src, num_layers) ? (LDense *)cl : NULL;
    LResize *layer_src_resize = found_resize(m_src, l_src, num_layers) ? (LResize *)cl : NULL;
    LConcat *layer_src_concat = found_concat(m_src, l_src, num_layers) ? (LConcat *)cl : NULL;
    LClamp *layer_src_clamp = found_clamp(m_src, l_src, num_layers) ? (LClamp *)cl : NULL;
    LMult *layer_src_mult = found_mult(m_src, l_src, num_layers) ? (LMult *)cl : NULL;
    LDiv *layer_src_div = found_div(m_src, l_src, num_layers) ? (LDiv *)cl : NULL;
    LActivation *layer_src_softplus = found_softplus(m_src, l_src, num_layers) ? (LActivation *)cl : NULL;
    LActivation *layer_src_tanh = found_tanh(m_src, l_src, num_layers) ? (LActivation *)cl : NULL;
    LExpand *layer_src_expand = found_expand(m_src, l_src, num_layers) ? (LExpand *)cl : NULL;
    LSelect *layer_src_select = found_select(m_src, l_src, num_layers) ? (LSelect *)cl : NULL;
    LDiff *layer_src_diff = found_diff(m_src, l_src, num_layers) ? (LDiff *)cl : NULL;
    LExp *layer_src_exp = found_exp(m_src, l_src, num_layers) ? (LExp *)cl : NULL;
    LConstOfTensor *layer_src_constoft = found_constoft(m_src, l_src, num_layers) ? (LConstOfTensor *)cl : NULL;
    LDropout *layer_src_dropout = found_dropout(m_src, l_src, num_layers) ? (LDropout *)cl : NULL;
    LActivation *layer_src_leakyrelu = found_leakyrelu(m_src, l_src, num_layers) ? (LActivation *)cl : NULL;
    LPad *layer_src_pad = found_pad(m_src, l_src, num_layers) ? (LPad *)cl : NULL;

    Layer *fpga_parent;
    if (!found_input(m_src, l_src, num_layers) && !found_constoft(m_src, l_src, num_layers)) {
      fpga_parent = fn_get_associated_layer(cl->parent[0], 0, &dummy);
      if (fpga_parent == NULL) fpga_parent = fn_get_associated_layer(cl->parent[0], 1, &dummy);
    }

    #ifdef FPGA_DEBUG
    if (found_input(m_src, l_src, num_layers))      printf("%3d: I\n", l_dst);
    if (found_maxp(m_src, l_src, num_layers))       printf("%3d: Maxpool : prev %d\n", l_dst, dummy);
    if (found_avgp(m_src, l_src, num_layers))       printf("%3d: Avgpool : prev %d\n", l_dst, dummy);
    if (found_upsampling(m_src, l_src, num_layers)) printf("%3d: Upsampling : prev %d\n", l_dst, dummy);
    if (found_linear(m_src, l_src, num_layers))     printf("%3d: Linear    : prev %d\n", l_dst, dummy);
    if (found_conv_cpu(m_src, l_src, num_layers))   printf("%3d: Conv CPU  : prev %d\n", l_dst, dummy);
    if (found_relu(m_src, l_src, num_layers))       printf("%3d: RELU      : prev %d\n", l_dst, dummy);
    if (found_bn(m_src, l_src, num_layers))         printf("%3d: BN        : prev %d\n", l_dst, dummy);
    if (found_reshape(m_src, l_src, num_layers))    printf("%3d: RESHAPE   : prev %d\n", l_dst, dummy);
    if (found_dense(m_src, l_src, num_layers))      printf("%3d: DENSE     : prev %d\n", l_dst, dummy);
    if (found_resize(m_src, l_src, num_layers))     printf("%3d: RESIZE    : prev %d\n", l_dst, dummy);
    if (found_softmax(m_src, l_src, num_layers))    printf("%3d: SOFTMAX   : prev %d\n", l_dst, dummy);
    if (found_sigmoid(m_src, l_src, num_layers))    printf("%3d: SIGMOID   : prev %d\n", l_dst, dummy);
    if (found_permute(m_src, l_src, num_layers))    printf("%3d: PERMUTE   : prev %d\n", l_dst, dummy);
    if (found_concat(m_src, l_src, num_layers))     printf("%3d: CONCAT    : prev %d\n", l_dst, dummy);
    if (found_clamp(m_src, l_src, num_layers))      printf("%3d: CLAMP     : prev %d\n", l_dst, dummy);
    if (found_mult(m_src, l_src, num_layers))       printf("%3d: MULT      : prev %d\n", l_dst, dummy);
    if (found_div(m_src, l_src, num_layers))        printf("%3d: DIV       : prev %d\n", l_dst, dummy);
    if (found_softplus(m_src, l_src, num_layers))   printf("%3d: SOFTPLUS  : prev %d\n", l_dst, dummy);
    if (found_tanh(m_src, l_src, num_layers))       printf("%3d: TANH      : prev %d\n", l_dst, dummy);
    if (found_expand(m_src, l_src, num_layers))     printf("%3d: EXPAND    : prev %d\n", l_dst, dummy);
    if (found_select(m_src, l_src, num_layers))     printf("%3d: SELECT    : prev %d\n", l_dst, dummy);
    if (found_diff(m_src, l_src, num_layers))       printf("%3d: DIFF      : prev %d\n", l_dst, dummy);
    if (found_exp(m_src, l_src, num_layers))        printf("%3d: EXP       : prev %d\n", l_dst, dummy);
    if (found_constoft(m_src, l_src, num_layers))   printf("%3d: CONSTOFT  : prev %d\n", l_dst, dummy);
    if (found_dropout(m_src, l_src, num_layers))    printf("%3d: DROPOUT   : prev %d\n", l_dst, dummy);
    if (found_leakyrelu(m_src, l_src, num_layers))  printf("%3d: LEAKYRELU : prev %d\n", l_dst, dummy);
    if (found_pad(m_src, l_src, num_layers))        printf("%3d: PAD       : prev %d\n", l_dst, dummy);
    #endif

    if (found_input(m_src, l_src, num_layers)) prev_layer = Input({cl->input->shape[1],cl->input->shape[2],cl->input->shape[3]});
    if (found_maxp(m_src, l_src, num_layers)) { 
      if (layer_src_maxp->pd->padding =="custom") prev_layer = new LMaxPool(fpga_parent, layer_src_maxp->pd->ksize, layer_src_maxp->pd->stride, layer_src_maxp->pd->pad, "", DEV_CPU, 0);
      else prev_layer = new LMaxPool(fpga_parent, layer_src_maxp->pd->ksize, layer_src_maxp->pd->stride, layer_src_maxp->pd->padding, "", DEV_CPU, 0);
    }
    if (found_avgp(m_src, l_src, num_layers)) { 
      if (layer_src_avgp->pd->padding =="custom") prev_layer = new LAveragePool(fpga_parent, layer_src_avgp->pd->ksize, layer_src_avgp->pd->stride, layer_src_avgp->pd->pad, "", DEV_CPU, 0);
      else prev_layer = new LAveragePool(fpga_parent, layer_src_avgp->pd->ksize, layer_src_avgp->pd->stride, layer_src_avgp->pd->padding, "", DEV_CPU, 0);
    }
    if (found_upsampling(m_src, l_src, num_layers)) prev_layer = UpSampling(fpga_parent, layer_src_upsampling->size, layer_src_upsampling->interpolation);
    if (found_linear(m_src, l_src, num_layers)) prev_layer = Linear(fpga_parent, layer_src_linear->params[0]);
    if (found_conv_cpu(m_src, l_src, num_layers)) prev_layer = new LConv(fpga_parent, layer_src_conv->cd->filters, layer_src_conv->cd->kernel_size, layer_src_conv->cd->strides, layer_src_conv->cd->padding,
                                                                                      layer_src_conv->cd->pads, layer_src_conv->cd->groups, layer_src_conv->cd->dilation_rate, layer_src_conv->cd->use_bias,
                                                                                      "",DEV_CPU, layer_src_conv->cd->mem_level);
    if (found_relu(m_src, l_src, num_layers)) prev_layer = ReLu(fpga_parent);
    if (found_reshape(m_src, l_src, num_layers)) {
      long int elements = 1;
      for (int i = 1; i < layer_src_reshape->ls.size(); i++) elements = elements * layer_src_reshape->ls[i];
      if (layer_src_reshape->ls[1] == elements && layer_src_reshape->ls.size() < 3 ) prev_layer = Reshape(fpga_parent, { -1 });
      else {
        vector<int> shape;
        for (int i = 1; i < layer_src_reshape->ls.size(); i++) shape.push_back(layer_src_reshape->ls[i]);
        prev_layer = Reshape(fpga_parent, shape);
      }
    }
    if (found_bn(m_src, l_src, num_layers)) prev_layer =  BatchNormalization(fpga_parent, layer_src_bn->momentum, layer_src_bn->epsilon, layer_src_bn->affine, "");
    if (found_dense(m_src, l_src, num_layers)) prev_layer = Dense(fpga_parent, layer_src_dense->ndim);
    if (found_resize(m_src, l_src, num_layers)) prev_layer = new LResize(fpga_parent, layer_src_resize->new_shape, layer_src_resize->reshape, layer_src_resize->da_mode,
                                                                         layer_src_resize->cval, layer_src_resize->coordinate_transformation_mode, layer_src_resize->name, layer_src_resize->dev, 0);
    if (found_softmax(m_src, l_src, num_layers)) prev_layer = Softmax(fpga_parent);
    if (found_sigmoid(m_src, l_src, num_layers)) prev_layer = Sigmoid(fpga_parent);
    if (found_permute(m_src, l_src, num_layers)) {
      vector<int> dims;
      dims.push_back(2);
      dims.push_back(1);
      dims.push_back(0);
      prev_layer = Permute(fpga_parent, dims);
    }
    if (found_clamp(m_src, l_src, num_layers)) prev_layer = Clamp(fpga_parent, layer_src_clamp->min, layer_src_clamp->max);

    if (found_mult(m_src, l_src, num_layers)) {
      if(layer_src_mult->parent.size() < 2) {
        prev_layer = Mult(fpga_parent, layer_src_mult->val);
      } else if(layer_src_mult->parent.size() == 2) {
        vector<Layer *> parent;
        Layer *fpga_parent;
        fpga_parent = fn_get_associated_layer(layer_src_mult->parent[0], 0, &dummy);
        parent.push_back(fpga_parent);
        fpga_parent = fn_get_associated_layer(layer_src_mult->parent[1], 0, &dummy1);
        parent.push_back(fpga_parent);
        vector<Layer *> operators = expand_broadcast(parent);
        prev_layer = Mult(operators[0], operators[1]);
      } else  msg("Error, Mult layer is only supported in FPGA with one or two parents","Model_for_fpga");
    }

    if (found_div(m_src, l_src, num_layers)) prev_layer = Div(fpga_parent, layer_src_div->val);
    if (found_softplus(m_src, l_src, num_layers)) prev_layer = Softplus(fpga_parent);
    if (found_tanh(m_src, l_src, num_layers)) prev_layer = Tanh(fpga_parent);
    if (found_expand(m_src, l_src, num_layers)) prev_layer = Expand(fpga_parent, layer_src_expand->size, "");
    if (found_select(m_src, l_src, num_layers)) prev_layer = Slice(fpga_parent, layer_src_select->sd->indices, "");
    if (found_diff(m_src, l_src, num_layers)) {
      if (layer_src_diff->left) {
        prev_layer = new LDiff(fpga_parent, layer_src_diff->val, "", DEV_CPU, layer_src_diff->mem_level);
      } else {
        prev_layer = new LDiff(layer_src_diff->val, fpga_parent, "", DEV_CPU, layer_src_diff->mem_level);
      }
    }
    if (found_exp(m_src, l_src, num_layers)) prev_layer = Exp(fpga_parent);
    if (found_constoft(m_src, l_src, num_layers)) prev_layer = ConstOfTensor(layer_src_constoft->const_tensor, layer_src_constoft->name);
    if (found_dropout(m_src, l_src, num_layers)) prev_layer = Dropout(fpga_parent, layer_src_dropout->df, layer_src_dropout->iw, layer_src_dropout->name);
    if (found_leakyrelu(m_src, l_src, num_layers)) {
      prev_layer = LeakyReLu(fpga_parent, layer_src_leakyrelu->params[0], "");
    }
    if (found_pad(m_src, l_src, num_layers)) prev_layer = Pad(fpga_parent, layer_src_pad->padding, layer_src_pad->constant, "");

    if (found_concat(m_src, l_src, num_layers)) {
      // dst parent layer
      vector<Layer *> parent;
      vector<int> dummy_vect;
      // the input format can be either GHWC or NCWH, but only one of them
      // we need to select which one
      int format = 0;
      if (fn_get_associated_layer(layer_src_concat->parent[0], format, &dummy) == NULL) format = 1;
      for(int p = 0; p < cl->parent.size();p++) {
        int dummy_el;
        parent.push_back(fn_get_associated_layer(layer_src_concat->parent[p], format, &dummy_el));
        dummy_vect.push_back(dummy_el);
      }
      //
      prev_layer = Concat(parent, layer_src_concat->axis, "");
    }


    if (found_concat(m_src, l_src, num_layers)) {
      // The concat layer is in the same format as its parent
      int format = 0;
      if (fn_get_associated_layer(cl->parent[0], format, &dummy) == NULL) format = 1;
      int format2;
      if (format == 0) format2 = chw_format; else format2 = ghwcpi_format;
      fn_set_associated_layer(cl, prev_layer, format, format2, cpu_device, l_dst);
    } else {
      fn_set_associated_layer(cl, prev_layer, 0, chw_format, cpu_device, l_dst);
    }
    associated_source_layer[l_dst].src = cl;
    associated_source_layer[l_dst].dst = prev_layer;
    associated_source_layer[l_dst].conv = NULL;
    associated_source_layer[l_dst].bn = NULL;
    associated_source_layer[l_dst].dense = NULL;
    l_dst++;


 } else if (found_add(m_src, l_src, num_layers)) {
    vector<Layer *> parent;
    vector<int> dummy_vect;
    for(int p = 0; p < cl->parent.size();p++) {
      int dummy_el;
      parent.push_back(fn_get_associated_layer(cl->parent[p], 0, &dummy_el));
      dummy_vect.push_back(dummy_el);
    }
    #ifdef FPGA_DEBUG
    printf("%3d: ADD        : prevs", l_dst);
    for(int p = 0; p < dummy_vect.size();p++){
      printf("%d ", dummy_vect[p]);
    }
    printf("\n");
    #endif
    prev_layer = Add(parent);
    fn_set_associated_layer(cl, prev_layer, 0, chw_format, cpu_device, l_dst);
    associated_source_layer[l_dst].src = cl;
    associated_source_layer[l_dst].dst = prev_layer;
    associated_source_layer[l_dst].conv = NULL;
    associated_source_layer[l_dst].bn = NULL;
    associated_source_layer[l_dst].dense = NULL;
    l_dst++;    
  } else {
    cout << "searching " << cl->name << "\n";
    msg("Error, unidentified layer","Model_for_fpga");
    exit(1);
  }

  int ind;

  l_src += num_layers_fused;
  }

  // now we create the input and output layers in the correct order
  for(int lin = 0; lin < m_src->lin.size(); lin++) first.push_back(fn_get_associated_layer(m_src->lin[lin], 0, &dummy));
  for(int lout = 0; lout < m_src->lout.size(); lout++) {
    Layer *l = fn_get_associated_layer(m_src->lout[lout], 0, &dummy);
    int device = fn_get_associated_layer_device(m_src->lout[lout]);
    if (device == fpga_device) {
      // The output layer runs on the FPGA, so we need to add a Transform layer and set it as the output layer
      Layer *parent_layer = fn_get_associated_layer(m_src->lout[lout], 1, &dummy);
      #ifdef FPGA_DEBUG
      printf("%3d: TRANSFORM  : prev %d\n", l_dst, dummy);
      #endif
      int copy_cpu_to_fpga = 0;
      int copy_fpga_to_cpu = 1;
      int format = fn_get_associated_layer_format(m_src->lout[lout]);
      int transform;
      if (format == chw_format) transform = 0; else transform = 1;
      l = Transform(parent_layer, copy_cpu_to_fpga, copy_fpga_to_cpu, transform, 0);
      fn_set_associated_layer(parent_layer, l, 1, ghwcpi_format, cpu_device, l_dst);
      l_dst++;
    }
    last.push_back(l);
  }

#ifdef FPGA_DEBUG
  printf("input Layers \n"); for(int lin = 0; lin < first.size(); lin++) cout << first[lin]->name << "\n";
  printf("Output Layers \n"); for(int lout = 0; lout < last.size(); lout++) cout << last[lout]->name << "\n";
  printf("End parsing/creating new network\n");
#endif

  // now we create the model
  net = Model({ first }, { last });
  build(net); 
  #ifdef FPGA_DEBUG
  summary(net);
  #endif

#ifdef FPGA_DEBUG
  // we list the whole FPGA model
  printf("-----------------------------------\n");
  printf("Layers (name, address, and its parents):\n");
  l=0;
  while (l < l_dst) {
    Layer *cl = net->layers[l];
    cout << "Layer " << l << " name: " << cl->name << " address: " << cl << " parents: ";
    for(int p = 0; p < cl->parent.size();p++){
      cout << cl->parent[p] << " ";
    }
    cout << "\n";
    l++;
  }
#endif

#ifdef FPGA_DEBUG
    printf("FIN MODEL\n");
#endif
  //get_fpga_model_params(net);

  // now we adapt the parameters (filter and bias for convs and vectors of normalization layers)
  for (int l=0; l<l_dst; l++) {
    Layer *cl = net->layers[l];
    #ifdef FPGA_DEBUG
    printf("Layer %d \n", l);
    #endif
    if (LConv *conv = dynamic_cast<LConv*>(cl)) { 
      LConv *layer_dst = (LConv *) net->layers[l];
      LConv *layer_src = (LConv *) fn_get_cpu_equivalent_layer(layer_dst, l_dst);

      #ifdef FPGA_DEBUG
      cout << "LConv adapting parameters for layer " << l<<" "<<layer_dst->name<<" (associated layer "<< layer_src->name<<")\n";
      #endif

      // filter
      collectTensor(layer_src, "param", 0);
      if(layer_src->cd->K->size != layer_dst->cd->K->size) tensor_padded(layer_src->cd->K, layer_dst->cd->K);
      Tensor::copy(layer_src->cd->K, layer_dst->cd->K);
      distributeTensor(layer_dst, "param", 0);
      //distribute filter to cpu
      Layer *sl=nullptr;
      Net *sn=layer_dst->net;
      for(int j=0;j<sn->snets[0]->layers.size();j++)
      if (sn->snets[0]->layers[j]->orig==layer_dst) {
          sl=sn->snets[0]->layers[j];
          break;
      }

      // bias
      collectTensor(layer_src, "param", 1);
      if (layer_src->cd->bias->size != layer_dst->cd->bias->size) tensor_padded(layer_src->cd->bias, layer_dst->cd->bias);
      else Tensor::copy(layer_src->cd->bias, layer_dst->cd->bias);
      distributeTensor(layer_dst, "param", 1);

      //distribute bias to cpu
      sn=layer_dst->net;
      for(int j=0;j<sn->snets[0]->layers.size();j++)
      if (sn->snets[0]->layers[j]->orig==layer_dst) {
          sl=sn->snets[0]->layers[j];
          break;
      }

      //copy to cpu memory
      //cpu_copy(layer_dst->params[0],sl->params[0]); //filter
      //cpu_copy(layer_dst->params[1],sl->params[1]); //bias

    } else if (LBatchNorm *bn = dynamic_cast<LBatchNorm*>(cl)) { 
      LBatchNorm *layer_dst = (LBatchNorm *) net->layers[l];
      LBatchNorm *layer_src = (LBatchNorm *) fn_get_cpu_equivalent_layer(layer_dst, l_dst);

      #ifdef FPGA_DEBUG
      cout << "LBatchNormalization adapting parameters for layer " << l<<" "<<layer_dst->name<<" (associated layer "<< layer_src->name<<")\n";
      #endif

      // filter
      collectTensor(layer_src, "param", 0);
      collectTensor(layer_src, "param", 1);
      collectTensor(layer_src, "param", 2);
      collectTensor(layer_src, "param", 3);
      Tensor::copy(layer_src->mean, layer_dst->mean);
      Tensor::copy(layer_src->variance, layer_dst->variance);
      Tensor::copy(layer_src->bn_g, layer_dst->bn_g);
      Tensor::copy(layer_src->bn_b, layer_dst->bn_b);
      distributeTensor(layer_dst, "param", 0);
      distributeTensor(layer_dst, "param", 1);
      distributeTensor(layer_dst, "param", 2);
      distributeTensor(layer_dst, "param", 3);

      //distribute filter to cpu
      /*Layer *sl=nullptr;
      Net *sn=layer_dst->net;
      for(int j=0;j<sn->snets[0]->layers.size();j++)
      if (sn->snets[0]->layers[j]->orig==layer_dst) {
          sl=sn->snets[0]->layers[j];
          break;
      }*/
      //copy to cpu memory
      //cpu_copy(layer_dst->params[0],sl->params[0]); 
      //cpu_copy(layer_dst->params[1],sl->params[1]); 
      //cpu_copy(layer_dst->params[2],sl->params[2]); 
      //cpu_copy(layer_dst->params[3],sl->params[3]); 

    } else if (LHLSinf *dl = dynamic_cast<LHLSinf *>(cl)) {
      LHLSinf *layer_dst = (LHLSinf *) net->layers[l]; 
      LConv *layer_src_conv = (LConv *) fn_get_cpu_equivalent_conv_layer(layer_dst, l_dst);
      LDense *layer_src_dense = (LDense *) fn_get_cpu_equivalent_dense_layer(layer_dst, l_dst);
      LBatchNorm *layer_src_bn = (LBatchNorm *) fn_get_cpu_equivalent_bn_layer(layer_dst, l_dst);

      #ifdef FPGA_DEBUG
      cout << "LHLSinf adapting parameters for layer" << l << " (" << layer_dst->name << ")\n";
      if (layer_src_conv != NULL) cout << "   associated conv layer " << layer_src_conv->name << "\n";
      if (layer_src_bn != NULL)   cout << "   associated bn layer " << layer_src_bn->name << "\n";
      if (layer_src_dense != NULL) cout << "   associated dense layer " << layer_src_dense->name << "\n";
      #endif

      if (layer_src_conv != NULL) {
        // conv filter
        collectTensor(layer_src_conv, "param", 0);
        filter_IHW_to_GIHWCPI(layer_src_conv->cd->K, layer_dst->filter);
        distributeTensor(layer_dst, "param", 0);
        // conv bias
        collectTensor(layer_src_conv, "param", 1);
        if (layer_src_conv->cd->use_bias) tensor_padded(layer_src_conv->cd->bias, layer_dst->bias); else memset(layer_dst->bias->ptr, 0, sizeof(float) * layer_dst->bias->size);
        distributeTensor(layer_dst, "param", 1);
      }

      if (layer_src_bn != NULL) {
        // batch norm
        collectTensor(layer_src_bn, "param", 0);
        collectTensor(layer_src_bn, "param", 1);
        collectTensor(layer_src_bn, "param", 2);
        collectTensor(layer_src_bn, "param", 3);
        get_batch_norm_values(layer_src_conv->cd->O->shape[1], layer_src_bn->mean, layer_src_bn->variance, layer_src_bn->bn_g, layer_src_bn->bn_b, layer_dst->batch_norm_values);
        distributeTensor(layer_dst, "param", 2);
      }

      if (layer_src_dense != NULL) {
        // dense params
        collectTensor(layer_src_dense, "param", 0);
        dense_to_conv(layer_src_dense->W->ptr, layer_src_dense->W->shape[0], layer_src_dense->W->shape[1], layer_dst->filter->ptr, layer_dst->Ichannels, layer_dst->Ochannels, layer_dst->KH, layer_dst->KW);
        if (hlsinf_filter_format == HLSINF_FP32) {
          layer_dst->filter->fpga_ptr = fpga_create_memory(layer_dst->filter->size*sizeof(float));  
          fpga_copy_memory_to_fpga(layer_dst->filter->ptr, (cl::Buffer *)layer_dst->filter->fpga_ptr, layer_dst->filter->size*sizeof(float));
        } else if (hlsinf_filter_format == HLSINF_API8) {
          layer_dst->filter->fpga_ptr = fpga_create_memory(layer_dst->filter->size);  
          fpga_copy_memory_to_fpga_and_format(layer_dst->filter->ptr, (cl::Buffer *)layer_dst->filter->fpga_ptr, layer_dst->filter->size, HLSINF_FP32, HLSINF_API8);
        } else {
          printf("Error (HLSinf forward), filter format not supported\n");
          exit(1);
        }

        distributeTensor(layer_dst, "param", 0);
        // dense bias
       	if (layer_src_dense->use_bias) {
          collectTensor(layer_src_dense, "param", 1);
          tensor_padded(layer_src_dense->bias, layer_dst->bias);
          if (hlsinf_bias_format == HLSINF_FP32) {
            layer_dst->bias->fpga_ptr = fpga_create_memory(layer_dst->bias->size*sizeof(float));  
            fpga_copy_memory_to_fpga(layer_dst->bias->ptr, (cl::Buffer *)layer_dst->bias->fpga_ptr, layer_dst->bias->size*sizeof(float));
          } else if (hlsinf_bias_format == HLSINF_API32) {
            layer_dst->bias->fpga_ptr = fpga_create_memory(layer_dst->bias->size*4);  
            fpga_copy_memory_to_fpga_and_format(layer_dst->bias->ptr, (cl::Buffer *)layer_dst->bias->fpga_ptr, layer_dst->bias->size, HLSINF_FP32, HLSINF_API32);
          } else {
            printf("Error (HLSinf forward), bias format not supported\n");
            exit(1);
          }
          distributeTensor(layer_dst, "param", 1);
	      }
      }

    } else if (LDense *dl = dynamic_cast<LDense *>(cl)) {
      LDense *layer_dst = (LDense *) net->layers[l]; 
      LDense *layer_src = (LDense *) fn_get_cpu_equivalent_layer(layer_dst,l_dst);

      #ifdef FPGA_DEBUG
      cout << "LDense adapting parameters for layer " << l<<" "<<layer_dst->name<<" (associated layer "<< layer_src->name<<")\n";
      #endif
      //w
      collectTensor(layer_src, "param", 0);
      tensor_padded(layer_src->W, layer_dst->W);
      distributeTensor(layer_dst, "param", 0);
            
      //bias
      collectTensor(layer_src, "param", 1);
      tensor_padded(layer_src->bias, layer_dst->bias);
      distributeTensor(layer_dst, "param", 1);
    } else if (LDropout *dl = dynamic_cast<LDropout *>(cl)) {
      LDropout *layer_dst = (LDropout *) net->layers[l];
      LDropout *layer_src = (LDropout *) fn_get_cpu_equivalent_layer(layer_dst, l_dst);

      #ifdef FPGA_DEBUG
      cout << "LDropout adapting mask for layer " << l<<" "<<layer_dst->name<<" (associated layer "<< layer_src->name<<")\n";
      #endif
      // mask
      tensor_padded(layer_src->mask, layer_dst->mask);
    }
    }

    #ifdef FPGA_DEBUG
    printf("End adapting parameters\n");
    #endif
    return net;
  #else
    msg("Error, FPGA is not activated","model_for_fpga");
  #endif
    }  
}//namespace

#else
#include "eddl/apis/eddl.h"
namespace eddl {
model toFPGA(model m_src, int kernel_version, int kernel_subversion) {
  msg("toFPGA only available for FPGA compilation","toFPGA");
  exit(1);
  return NULL;
}
}
#endif