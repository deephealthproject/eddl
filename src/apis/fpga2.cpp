/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: December 2021
* Author: GAP Research Group (UPV), contact: jflich@disca.upv.es
* All rights reserved
*/


// TODO: Mult layer con mas de dos parámetros no está bien soportado
// TODO: Layers Add y Sum son lo mismo?

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

////////////////////////////////////////////////////////
///// EDDL is a wrapper class to ease and define the API
////////////////////////////////////////////////////////

using namespace std;

namespace eddl {


#define LAYER_NONE 0
#define LAYER_RELU    1
#define LAYER_LEAKY_RELU 2
#define LAYER_SOFTMAX 3
#define LAYER_SIGMOID 4
#define LAYER_SOFTPLUS 5
#define LAYER_TANH 6
#define LAYER_LINEAR 7
#define LAYER_MAX_POOL 8
#define LAYER_AVG_POOL 9
#define LAYER_RESHAPE 10
#define LAYER_RESIZE 11
#define LAYER_DENSE 12
#define LAYER_CONCAT 13
#define LAYER_EXPAND 14
#define LAYER_SELECT 15
#define LAYER_MULT 16
#define LAYER_DIV 17
#define LAYER_DIFF 18
#define LAYER_EXP 19
#define LAYER_PERMUTE 20
#define LAYER_ADD 21
#define LAYER_SUM 22
#define LAYER_CONST_OF_TENSOR 23
#define LAYER_CLAMP 24
#define LAYER_PAD 25
#define LAYER_UPSAMPLING 26
#define LAYER_BATCH_NORM 27
#define LAYER_DROP_OUT 28
#define LAYER_CONV 29
#define LAYER_HLSINF 30
#define LAYER_INPUT 31
#define LAYER_TRANSFORM 32
#define LAYER_DUMMY_CONCAT 33

#define MAX_LAYERS 1000
#define MAX_BUFFERS 500
#define MAX_UNIFIED_BUFFERS 10
#define MAX_CHILDS    5
#define MAX_READ_LAYERS 10
#define MAX_WRITE_LAYERS 10

int CPI;
int CPO;
int graph_size;
int num_buffers;
int num_unified_buffers;

struct {
  int layer_type;              // type of layer
  int in_layer;                // whether this is an input layer for the model
  int out_layer;               // whether this is an output layer for the model
  int i, o;                    // number of input and output channels
  int hi, wi, ho, wo;          // height and width of input and output channels
  int kh, kw;                  // filter height and width
  int pt, pb, pl, pr;          // padding (top, bottom, left, right)
  int sh, sw;                  // stride (height and width = vertical and horizontal))
  int input_offset;            // offset at the input for the HLSinf accelerator
  int output_offset;           // offset at the output for the HLSinf accelerator
  float relu_factor;           // ReLu factor (Leaky ReLu)
  int input_buffer[2];         
  int output_buffer;
  char sz_layer_name[100];
  int next[MAX_CHILDS];
  int prev[3];
  Layer *layer;                // pointer to layer in the original model
  Layer *final_layer;          // pointer to the layer in the final model
  LConv *conv_layer;           // pointer to the conv layer in the original model
  LBatchNorm *bn_layer;        // pointer to the BN layer in the original model
  LDense *dense_layer;         // pointer to the Dense layer in the original model
  Tensor *input_tensor;        
  Tensor *output_tensor;
  Tensor *weight_tensor;
  Tensor *bias_tensor;
  Tensor *add_tensor;
  Tensor *bn_mean_tensor;
  Tensor *bn_var_tensor;
  Tensor *bn_bn_g_tensor;
  Tensor *bn_bn_b_tensor;
  int apply_conv;
  int apply_relu;
  int apply_maxp;
  int apply_avgp;
  int apply_add;
  int apply_add_relu;
  int apply_bn;
  int apply_bn_relu;
  float bn_relu_factor;
  int apply_cliping;
  int min_clip;
  int max_clip;
  int apply_shift;
  int pos_shift;
  int dir_shift;
  int apply_stm;
  int upscale_factor;
  int apply_dense;
  int apply_weight_buffer;
  int first_row_weight_buffer;
  int cpu2fpga;
  int fpga2cpu;
  int transform;
  int column;
  int visited;
  int output_fpga_buffer_created;
} layer_graph[MAX_LAYERS];

struct {
  int size;                                 // size of the buffer to allocate
  void *ptr;                                // pointer to the allocated buffer
  int write_layers[MAX_WRITE_LAYERS];       // up to three write layers (layers that write on the buffer)
  int write_offset[MAX_WRITE_LAYERS];       // write offsets for each write layer
  int read_layers[MAX_READ_LAYERS];         // up to two read layrs (layers that read from the buffer)
  int read_offset[MAX_READ_LAYERS];         // read offsets for each read layer
} buffers[MAX_BUFFERS];

struct {
  int valid;
  void *ptr;
  int num_buffers;
  int new_buffer;
  int old_buffer[3];
  int from[3];
  int to[3];
  int size[3];
} unified_buffers[MAX_UNIFIED_BUFFERS];

int shared_output_buffer(int buffer, int *offset, int *entry, int *total_size) {
  for (int x=0; x < num_unified_buffers; x++) {
    if (unified_buffers[x].valid) {
      *total_size = 0;
      for (int p=0; p<unified_buffers[x].num_buffers; p++) if ((unified_buffers[x].to[p] + 1) > *total_size) *total_size = unified_buffers[x].to[p] + 1;
      for (int p=0; p<unified_buffers[x].num_buffers; p++) {
        if (unified_buffers[x].old_buffer[p] == buffer) {
          *offset = unified_buffers[x].from[p];
          *entry = x;
          return 1;
        }
      }
    }
  }
  return 0;
}

int get_layer_type(Layer *layer) {
  if (LInput *dl = dynamic_cast<LInput *>(layer)) return LAYER_INPUT;
  if (LActivation *dl = dynamic_cast<LActivation *>(layer)) if (dl->act == "relu") return LAYER_RELU;
  if (LActivation *dl = dynamic_cast<LActivation *>(layer)) if (dl->act == "leaky_relu") return LAYER_LEAKY_RELU;
  if (LActivation *dl = dynamic_cast<LActivation *>(layer)) if (dl->act == "softmax") return LAYER_SOFTMAX;
  if (LActivation *dl = dynamic_cast<LActivation *>(layer)) if (dl->act == "sigmoid") return LAYER_SIGMOID;
  if (LActivation *dl = dynamic_cast<LActivation *>(layer)) if (dl->act == "softplus") return LAYER_SOFTPLUS;
  if (LActivation *dl = dynamic_cast<LActivation *>(layer)) if (dl->act == "tanh") return LAYER_TANH;
  if (LActivation *dl = dynamic_cast<LActivation *>(layer)) if (dl->act == "linear") return LAYER_LINEAR;
  if (LMaxPool *dl = dynamic_cast<LMaxPool *>(layer)) return LAYER_MAX_POOL;
  if (LAveragePool *dl = dynamic_cast<LAveragePool *>(layer)) return LAYER_AVG_POOL;
  if (LReshape *dl = dynamic_cast<LReshape *>(layer)) return LAYER_RESHAPE;
  if (LResize *dl = dynamic_cast<LResize *>(layer)) return LAYER_RESIZE;
  if (LDense *dl = dynamic_cast<LDense *>(layer)) return LAYER_DENSE;
  if (LConcat *dl = dynamic_cast<LConcat *>(layer)) return LAYER_CONCAT;
  if (LExpand *dl = dynamic_cast<LExpand *>(layer)) return LAYER_EXPAND;
  if (LSelect *dl = dynamic_cast<LSelect *>(layer)) return LAYER_SELECT;
  if (LMult *dl = dynamic_cast<LMult *>(layer)) return LAYER_MULT;
  if (LDiv *dl = dynamic_cast<LDiv *>(layer)) return LAYER_DIV;
  if (LDiff *dl = dynamic_cast<LDiff *>(layer)) return LAYER_DIFF;
  if (LExp *dl = dynamic_cast<LExp *>(layer)) return LAYER_EXP;
  if (LPermute *dl = dynamic_cast<LPermute *>(layer)) return LAYER_PERMUTE;
  if (LAdd *dl = dynamic_cast<LAdd *>(layer)) return LAYER_ADD;
  if (LSum *dl = dynamic_cast<LSum *>(layer)) return LAYER_SUM;
  if (LConstOfTensor *dl = dynamic_cast<LConstOfTensor *>(layer)) return LAYER_CONST_OF_TENSOR;
  if (LClamp *dl = dynamic_cast<LClamp *>(layer)) return LAYER_CLAMP;
  if (LPad *dl = dynamic_cast<LPad *>(layer)) return LAYER_PAD;
  if (LUpSampling *dl = dynamic_cast<LUpSampling *>(layer)) return LAYER_UPSAMPLING;
  if (LBatchNorm *dl = dynamic_cast<LBatchNorm *>(layer)) return LAYER_BATCH_NORM;
  if (LDropout *dl = dynamic_cast<LDropout *>(layer)) return LAYER_DROP_OUT;
  if (LConv *dl = dynamic_cast<LConv *>(layer)) return LAYER_CONV;
  if (LTransform *dl = dynamic_cast<LTransform *>(layer)) return LAYER_TRANSFORM;
  printf("unrecognized layer type\n"); exit(1);
}

int get_i(Layer *layer) {
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->I->shape[1];
  return layer->input->shape[1];
  return 0;
}

int get_o(Layer *layer) {
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->O->shape[1];
  return layer->output->shape[1];
}

int get_hi(Layer *layer) {
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->I->shape[2];
  return layer->input->shape[2];
};

int get_wi(Layer *layer) {
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->I->shape[3];
  return layer->input->shape[3];
}

int get_ho(Layer *layer) {
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->O->shape[2];
  return layer->output->shape[2];
};

int get_wo(Layer *layer) {
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->O->shape[3];
  return layer->output->shape[3];
}

int get_kh(Layer *layer) {
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->kr;
  if (LMaxPool *maxp = dynamic_cast<LMaxPool *>(layer)) return maxp->pd->kr;
  if (LAveragePool *avgp = dynamic_cast<LAveragePool *>(layer)) return avgp->pd->kr;
  return 0;
}

int get_kw(Layer *layer) {
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->kc;
  if (LMaxPool *maxp = dynamic_cast<LMaxPool *>(layer)) return maxp->pd->kc;
  if (LAveragePool *avgp = dynamic_cast<LAveragePool *>(layer)) return avgp->pd->kc;
  return 0;
};

int get_pt(Layer *layer) {
  if (LPad *pad = dynamic_cast<LPad *>(layer)) return pad->padding[0];
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->padrt;
  return 0;
}

int get_pb(Layer *layer) {
  if (LPad *pad = dynamic_cast<LPad *>(layer)) return pad->padding[2];
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->padrb;
  return 0;
};

int get_pl(Layer *layer) {
  if (LPad *pad = dynamic_cast<LPad *>(layer)) return pad->padding[3];
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->padcl;
  return 0;
};

int get_pr(Layer *layer) {
  if (LPad *pad = dynamic_cast<LPad *>(layer)) return pad->padding[1];
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->padcr;
  return 0;
};

int get_sh(Layer *layer) {
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->sr;
  if (LMaxPool *maxp = dynamic_cast<LMaxPool *>(layer)) return maxp->pd->sr;
  if (LAveragePool *avgp = dynamic_cast<LAveragePool *>(layer)) return avgp->pd->sr;
  return 0;
};

int get_sw(Layer *layer) {
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->sc;
  if (LMaxPool *maxp = dynamic_cast<LMaxPool *>(layer)) return maxp->pd->kc;
  if (LAveragePool *avgp = dynamic_cast<LAveragePool *>(layer)) return avgp->pd->kc;
  return 0;
};

int get_input_offset(Layer *layer) {
  if (LSelect *sel = dynamic_cast<LSelect *>(layer)) {
    int first_dim0 = sel->sd->idxs_range[0][0];
    int last_dim0  = sel->sd->idxs_range[0][1];
    int first_dim1 = sel->sd->idxs_range[1][0];
    int last_dim1  = sel->sd->idxs_range[1][1];
    int first_dim2 = sel->sd->idxs_range[2][0];
    int last_dim2  = sel->sd->idxs_range[2][1];
    return first_dim0 * (last_dim1 - first_dim1 + 1) * (last_dim2 - first_dim2 + 1);
  }
  return 0;
}

float get_relu_factor(Layer *layer) {
  if (LActivation *dl = dynamic_cast<LActivation *>(layer)) if (dl->act == "leaky_relu") return dl->params[0];
  return 0;
}

int get_upscale_factor(Layer *layer) {
  if (LUpSampling *ups = dynamic_cast<LUpSampling *>(layer)) return ups->output->shape[2] / ups->input->shape[2];
  return 1;
}

Tensor *get_input_tensor(Layer *layer) {return layer->input;}
Tensor *get_output_tensor(Layer *layer) {return layer->output;}
Tensor *get_weight_tensor(Layer *layer) {
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->K;
  return nullptr;
}
Tensor *get_bias_tensor(Layer *layer) {
  if (LConv *conv = dynamic_cast<LConv *>(layer)) return conv->cd->bias;
  return nullptr;
}

Tensor *get_bn_mean_tensor(Layer *layer) {
  if (LBatchNorm *bn = dynamic_cast<LBatchNorm *>(layer)) return bn->mean;
  return nullptr;
}

Tensor *get_bn_var_tensor(Layer *layer) {
  if (LBatchNorm *bn = dynamic_cast<LBatchNorm *>(layer)) return bn->variance;
  return nullptr;
}

Tensor *get_bn_bn_g_tensor(Layer *layer) {
  if (LBatchNorm *bn = dynamic_cast<LBatchNorm *>(layer)) return bn->bn_g;
  return nullptr;
}

Tensor *get_bn_bn_b_tensor(Layer *layer) {
  if (LBatchNorm *bn = dynamic_cast<LBatchNorm *>(layer)) return bn->bn_b;
  return nullptr;
}

Tensor *get_add_tensor(Layer *layer) {return nullptr;}

int get_layer(Layer *layer) {
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer == layer) return l;
  }
  return -1;
}

int build_network_graph(model m) {
 
  int num_layers = m->layers.size();
  if (num_layers > MAX_LAYERS) {printf("Error, too many layers\n"); exit(1);}

  graph_size = 0;
  for (int l=0; l<num_layers; l++) {
    Layer *layer = m->layers[l];
    layer_graph[l].layer_type = get_layer_type(layer);
    layer_graph[l].i = get_i(layer);
    layer_graph[l].o = get_o(layer);
    layer_graph[l].hi = get_hi(layer);
    layer_graph[l].wi = get_wi(layer);
    layer_graph[l].ho = get_ho(layer);
    layer_graph[l].wo = get_wo(layer);
    layer_graph[l].kh = get_kh(layer);
    layer_graph[l].kw = get_kw(layer);
    layer_graph[l].pt = get_pt(layer);
    layer_graph[l].pb = get_pb(layer);
    layer_graph[l].pl = get_pl(layer);
    layer_graph[l].pr = get_pr(layer);
    layer_graph[l].sh = get_sh(layer);
    layer_graph[l].sw = get_sw(layer);
    layer_graph[l].input_offset = get_input_offset(layer);
    layer_graph[l].output_offset = 0;
    layer_graph[l].relu_factor = get_relu_factor(layer);
    layer_graph[l].layer = layer;
    layer_graph[l].input_buffer[0] = -1;
    layer_graph[l].input_buffer[1] = -1;
    layer_graph[l].output_buffer = -1;
    if (layer_graph[l].layer_type == LAYER_CONV) layer_graph[l].conv_layer = (LConv *)layer; else layer_graph[l].conv_layer = nullptr;
    if (layer_graph[l].layer_type == LAYER_BATCH_NORM) layer_graph[l].bn_layer = (LBatchNorm *)layer; else layer_graph[l].bn_layer = nullptr;
    if (layer_graph[l].layer_type == LAYER_DENSE) layer_graph[l].dense_layer = (LDense *)layer; else layer_graph[l].dense_layer = nullptr;
    layer_graph[l].input_tensor = get_input_tensor(layer);
    layer_graph[l].output_tensor = get_output_tensor(layer);
    layer_graph[l].weight_tensor = get_weight_tensor(layer);
    layer_graph[l].bias_tensor = get_bias_tensor(layer);    
    layer_graph[l].add_tensor = get_add_tensor(layer);
    layer_graph[l].bn_mean_tensor = get_bn_mean_tensor(layer);
    layer_graph[l].bn_var_tensor = get_bn_var_tensor(layer);
    layer_graph[l].bn_bn_g_tensor = get_bn_bn_g_tensor(layer);
    layer_graph[l].bn_bn_b_tensor = get_bn_bn_b_tensor(layer);
    layer_graph[l].upscale_factor = get_upscale_factor(layer);


    for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = -1;
    layer_graph[l].prev[0] = -1;
    layer_graph[l].prev[1] = -1;
    layer_graph[l].prev[2] = -1;
    if (l!=0) {
      for (int p=0; p<layer->parent.size(); p++) {
        int parent_layer = get_layer(layer->parent[p]);
        if (parent_layer == -1) {printf("error, parent not found (%p)\n", layer->parent[p]); exit(1);}
	layer_graph[l].prev[p] = parent_layer;
	int assigned = 0;
	for (int x=0; x<MAX_CHILDS; x++) if (layer_graph[parent_layer].next[x] == -1) {layer_graph[parent_layer].next[x] = l; assigned = 1; break;}
       
        if (!assigned) {printf("Error, no left next entries in a layer (parent layer %d, layer %d\n", parent_layer, l); exit(1);}
      }
    }
    graph_size++;
  }

  // in and out layers
  for (int lin = 0; lin < m->lin.size(); lin++) { 
    int l = get_layer(m->lin[lin]);
    if (l == -1) {printf("Error, layer not found\n"); exit(1);}
    layer_graph[l].in_layer = 1;
  }
  for(int lout = 0; lout < m->lout.size(); lout++) {
    int l = get_layer(m->lout[lout]);
    if (l == -1) {printf("Error, layer not found\n"); exit(1);}
    layer_graph[l].out_layer = 1;
  }

  return num_layers;
}

char *get_layer_name(int layer_type) {
  switch (layer_type) {
    case LAYER_NONE: return "None";
    case LAYER_RELU: return "ReLu";
    case LAYER_LEAKY_RELU: return "LeakyReLu";
    case LAYER_SOFTMAX: return "SoftMax";
    case LAYER_SIGMOID: return "Sigmoid";
    case LAYER_SOFTPLUS: return "SoftPlus";
    case LAYER_TANH: return "TanH";
    case LAYER_LINEAR: return "Linear";
    case LAYER_MAX_POOL: return "MaxPool";
    case LAYER_AVG_POOL: return "AvgPool";
    case LAYER_RESHAPE: return "Reshape";
    case LAYER_RESIZE: return "Resize";
    case LAYER_DENSE: return "Dense";
    case LAYER_CONCAT: return "Concat";
    case LAYER_EXPAND: return "Expand";
    case LAYER_SELECT: return "Select";
    case LAYER_MULT: return "Mult";
    case LAYER_DIV: return "Div";
    case LAYER_DIFF: return "Diff";
    case LAYER_EXP: return "Exp";
    case LAYER_PERMUTE: return "Permute";
    case LAYER_ADD: return "Add";
    case LAYER_SUM: return "Sum";
    case LAYER_CONST_OF_TENSOR: return "ConstOfT";
    case LAYER_CLAMP: return "Clamp";
    case LAYER_PAD: return "Pad";
    case LAYER_UPSAMPLING: return "UpSampling";
    case LAYER_BATCH_NORM: return "BatchNorm";
    case LAYER_DROP_OUT: return "DropOut";
    case LAYER_CONV: return "Conv";
    case LAYER_HLSINF: return "HLSinf";
    case LAYER_INPUT: return "Input";
    case LAYER_TRANSFORM: return "Transform";
    case LAYER_DUMMY_CONCAT: return "DConcat";
  }
  return "??????????";
}

void print_network_graph() {
#ifdef DEBUG_FPGA	
  printf("Network graph:\n");
  printf("                                                                         - HLSinf options  |\n");
  printf("                                                                         - R B B           |\n");
  printf("                       O                                                 - E N N           |\n");
  printf("                   I   U                             Pad.                - L   L           |\n");
  printf("LID - Layer      - N - T - .I. .O. .H. .W. - KH KW - TBLR - SH SW - ReLF - U   R           | Layer_ptr..... - Input......... - Output........ - Weight........ - Bias.......... - Add........... - BN_mean....... - BN_var........ - BN_bn_g....... - BN_bn_n....... - next...\n");  
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type != LAYER_NONE)
      printf("%3d - %-10s - %1d - %1d - %3d %3d %3d %3d - %2d %2d - %1d%1d%1d%1d - %2d %2d - %4.2f - %1d %1d %1d           -  %-14p - %-14p - %-14p - %-14p - %-14p - %-14p - %-14p - %-14p - %-14p - %-14p - %3d %3d %3d\n", l,   
                                                   get_layer_name(layer_graph[l].layer_type), 
                                                   layer_graph[l].in_layer, layer_graph[l].out_layer, layer_graph[l].i, layer_graph[l].o, layer_graph[l].h, layer_graph[l].w, layer_graph[l].kh, layer_graph[l].kw,
                                                   layer_graph[l].pt, layer_graph[l].pb, layer_graph[l].pl, layer_graph[l].pr, layer_graph[l].sh, layer_graph[l].sw, layer_graph[l].relu_factor, 
                                                   layer_graph[l].apply_relu, layer_graph[l].apply_bn, layer_graph[l].apply_bn_relu,
                                                   layer_graph[l].layer, layer_graph[l].input_tensor, layer_graph[l].output_tensor, layer_graph[l].weight_tensor, layer_graph[l].bias_tensor, layer_graph[l].add_tensor, 
                                                   layer_graph[l].bn_mean_tensor, layer_graph[l].bn_var_tensor, layer_graph[l].bn_bn_g_tensor, layer_graph[l].bn_bn_b_tensor, layer_graph[l].next[0], layer_graph[l].next[1], layer_graph[l].next[2]);
  }
#else
  printf("Network graph:\n");
  printf("                                                                                 | HLSinf options                                       | Transform |\n");
  printf("                                                                                 | C R M B B U                                          | c f t     |\n");
  printf("                       O                                                         | O E A N N P                                          | 2 2 r     |\n");
  printf("                   I   U                                     Pad.                | N L X   L S input     output   input  input  output  | f c a     |\n");
  printf("LID - Layer      - N - T - .I. HI. WI. .O. HO. RO. - KH KW - TBLR - SH SW - ReLF | V U P   R F offset    offset   buffer buffer buffer  |     n     | nexts.....  prevs.....\n");      
  printf("============================================================================================================================================================================\n");
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type != LAYER_NONE)
      printf("%3d - %-10s - %1d - %1d - %3d %3d %3d %3d %3d %3d - %2d %2d - %1d%1d%1d%1d - %2d %2d - %4.2f | %1d %1d %1d %1d %1d %1d %8d %8d %6d %6d %6d    | %1d %1d %1d     | %3d %3d %3d | %3d %3d %3d\n", l, 
		                                   get_layer_name(layer_graph[l].layer_type), 
						   layer_graph[l].in_layer, layer_graph[l].out_layer, layer_graph[l].i, layer_graph[l].hi, layer_graph[l].wi, layer_graph[l].o, layer_graph[l].ho, layer_graph[l].wo,
						   layer_graph[l].kh, layer_graph[l].kw,
                                                   layer_graph[l].pt, layer_graph[l].pb, layer_graph[l].pl, layer_graph[l].pr, layer_graph[l].sh, layer_graph[l].sw, layer_graph[l].relu_factor, 
						   layer_graph[l].apply_conv, layer_graph[l].apply_relu, layer_graph[l].apply_maxp, layer_graph[l].apply_bn, layer_graph[l].apply_bn_relu, layer_graph[l].upscale_factor, layer_graph[l].input_offset, layer_graph[l].output_offset,
						   layer_graph[l].input_buffer[0], layer_graph[l].input_buffer[1], layer_graph[l].output_buffer,
						   layer_graph[l].cpu2fpga, layer_graph[l].fpga2cpu, layer_graph[l].transform,
						   layer_graph[l].next[0], layer_graph[l].next[1], layer_graph[l].next[2], layer_graph[l].prev[0], layer_graph[l].prev[1], layer_graph[l].prev[2]);
  }
#endif
}

void update_prevs(int l_old, int l_new) {
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type != LAYER_NONE) {
      if (layer_graph[l].prev[0] == l_old) layer_graph[l].prev[0] = l_new;
      if (layer_graph[l].prev[1] == l_old) layer_graph[l].prev[1] = l_new;
      if (layer_graph[l].prev[2] == l_old) layer_graph[l].prev[2] = l_new;
    }
  }
}

void update_nexts(int l_old, int l_new) {
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type != LAYER_NONE) {
      for (int x=0; x<MAX_CHILDS; x++) {
        if (layer_graph[l].next[x] == l_old) {
          layer_graph[l].next[x] = l_new;
	  if ((layer_graph[l_new].prev[0] != l) && (layer_graph[l_new].prev[1] != l) && (layer_graph[l_new].prev[2] != l)) {
	    if (layer_graph[l_new].prev[0] == -1) layer_graph[l_new].prev[0] = l;
	    else if (layer_graph[l_new].prev[1] == -1) layer_graph[l_new].prev[1] = l;
	    else if (layer_graph[l_new].prev[2] == -1) layer_graph[l_new].prev[2] = l;
	    else {printf("Error, no slots available in prev field\n"); exit(1);}
          }
	}
      }
    }
  }
}

void apply_conv_relu() {

  if (!hlsinf_conv_support) return;
  if (!hlsinf_relu_support) return;

  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONV) {
      if ((layer_graph[l].wo <= hlsinf_wo_max) && (layer_graph[l].kh <= 3) && (layer_graph[l].kw <= 3)) {
        int nl = layer_graph[l].next[0];
        if (layer_graph[nl].layer_type == LAYER_RELU) {
          layer_graph[l].layer_type = LAYER_HLSINF;
	  strcpy(layer_graph[l].sz_layer_name, "HLSinf (conv + relu)");
          layer_graph[l].i = ceil((float)layer_graph[l].i/CPI) * CPI;
          layer_graph[l].o = ceil((float)layer_graph[l].o/CPO) * CPO;
    	  layer_graph[l].apply_conv = 1;
          layer_graph[l].apply_relu = 1;
          layer_graph[l].output_tensor = layer_graph[nl].output_tensor;
          layer_graph[nl].layer_type = LAYER_NONE;
          for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nl].next[x];
	  update_prevs(nl, l);
	  update_nexts(nl, l);
	}
      }
    }
  }
}

void apply_conv_add() {
  if (!hlsinf_conv_support) return;
  if (!hlsinf_add_support) return;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONV) {
      if ((layer_graph[l].wo <= hlsinf_wo_max) && (layer_graph[l].kh <= 3) && (layer_graph[l].kw <= 3)) {
        int nl = layer_graph[l].next[0];
        if (layer_graph[nl].layer_type == LAYER_ADD) {
          layer_graph[l].layer_type = LAYER_HLSINF;
          strcpy(layer_graph[l].sz_layer_name, "HLSinf (conv + add)");	  
          layer_graph[l].i = ceil((float)layer_graph[l].i/CPI) * CPI;
          layer_graph[l].o = ceil((float)layer_graph[l].o/CPO) * CPO;
          layer_graph[l].apply_conv = 1;
	  layer_graph[l].apply_add = 1;
          layer_graph[l].add_tensor = layer_graph[nl].add_tensor;
          layer_graph[l].output_tensor = layer_graph[nl].output_tensor;
          layer_graph[nl].layer_type = LAYER_NONE;
          for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nl].next[x];
          update_prevs(nl, l);
	  update_nexts(nl, l);
	}
      }
    }
  }
}

void apply_conv_maxp() {
  if (!hlsinf_conv_support) return;
  if (!hlsinf_maxp_support) return;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONV) {
      if ((layer_graph[l].wo <= hlsinf_wo_max) && (layer_graph[l].kh <= 3) && (layer_graph[l].kw <= 3)) {
        int nl = layer_graph[l].next[0];
        if (layer_graph[nl].layer_type == LAYER_MAX_POOL) {
          layer_graph[l].layer_type = LAYER_HLSINF;
          strcpy(layer_graph[l].sz_layer_name, "HLSinf (conv + maxp)");	  
          layer_graph[l].i = ceil((float)layer_graph[l].i/CPI) * CPI;
          layer_graph[l].o = ceil((float)layer_graph[l].o/CPO) * CPO;
	  layer_graph[l].ho = layer_graph[nl].ho;
	  layer_graph[l].wo = layer_graph[nl].wo;
          layer_graph[l].apply_conv = 1;
          layer_graph[l].apply_maxp = 1;
          layer_graph[l].output_tensor = layer_graph[nl].output_tensor;
          layer_graph[nl].layer_type = LAYER_NONE;
          for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nl].next[x];
          update_prevs(nl, l);
	  update_nexts(nl, l);
	}
      }
    }
  }
}

void apply_conv_mult_clamp_relu_maxp() {}
void apply_conv_mult_clamp_relu() {}
void apply_conv_mult() {}
void apply_conv_div_clamp_relu_maxp() {}
void apply_conv_div_clamp_relu() {}
void apply_conv_div() {}
void apply_conv_softplus_tanh_mult_add() {}
void apply_conv_softplus_tanh_mult() {}
void apply_pad_conv_sigmoid_tanh_maxp_add() {}
void apply_pad_conv_sigmoid_tanh_maxp() {}

void apply_pad_conv_relu_maxp() {
  if (!hlsinf_conv_support) return;
  if (!hlsinf_maxp_support) return;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_PAD) {
      int nl = layer_graph[l].next[0];
      if ((nl != -1) && (layer_graph[nl].layer_type == LAYER_CONV)) {
        if ((layer_graph[nl].wo <= hlsinf_wo_max) && (layer_graph[nl].kh <= 3) && (layer_graph[nl].kw <= 3)) {
          int nnl = layer_graph[nl].next[0];
          if ((nnl != -1) && (layer_graph[nnl].layer_type == LAYER_RELU)) {
	    int nnnl = layer_graph[nnl].next[0];
	    if ((nnnl != -1) && (layer_graph[nnnl].layer_type == LAYER_MAX_POOL)) {
              layer_graph[l].layer_type = LAYER_HLSINF;
              strcpy(layer_graph[l].sz_layer_name, "HLSinf (pad + conv + relu + maxp)");
              layer_graph[l].i = ceil((float)layer_graph[nl].i/CPI) * CPI;
              layer_graph[l].o = ceil((float)layer_graph[nl].o/CPO) * CPO;
              layer_graph[l].ho = layer_graph[nnnl].ho;
              layer_graph[l].wo = layer_graph[nnnl].wo;
              layer_graph[l].kh = layer_graph[nl].kh;
              layer_graph[l].kw = layer_graph[nl].kw;
              layer_graph[l].sh = layer_graph[nl].sh;
              layer_graph[l].sw = layer_graph[nl].sw;
              layer_graph[l].pt += layer_graph[nl].pt;
              layer_graph[l].pb += layer_graph[nl].pb;
              layer_graph[l].pl += layer_graph[nl].pl;
              layer_graph[l].pr += layer_graph[nl].pr;
              layer_graph[l].conv_layer = layer_graph[nl].conv_layer;
              layer_graph[l].output_tensor = layer_graph[nnnl].output_tensor;
              layer_graph[l].weight_tensor = layer_graph[nl].weight_tensor;
              layer_graph[l].bias_tensor = layer_graph[nl].bias_tensor;
              layer_graph[l].apply_conv = 1;
              layer_graph[l].apply_relu = 1;
              layer_graph[l].bn_relu_factor = layer_graph[nnl].relu_factor;
              layer_graph[l].apply_maxp = 1;
              layer_graph[nl].layer_type = LAYER_NONE;
              layer_graph[nnl].layer_type = LAYER_NONE;
              layer_graph[nnnl].layer_type = LAYER_NONE;
	      for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nnnl].next[x];
              update_prevs(nl, l);
              update_prevs(nnl, l);
	      update_prevs(nnnl, l);
	      update_nexts(nl, l);
	      update_nexts(nnl, l);
	      update_nexts(nnnl, l);
	    }
          }
        }
      }
    }
  }
}

void apply_pad_conv_relu() {
  if (!hlsinf_conv_support) return;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_PAD) {
      int nl = layer_graph[l].next[0];
      if ((nl != -1) && (layer_graph[nl].layer_type == LAYER_CONV)) {
        if ((layer_graph[nl].wo <= hlsinf_wo_max) && (layer_graph[nl].kh <= 3) && (layer_graph[nl].kw <= 3)) {
          int nnl = layer_graph[nl].next[0];
          if ((nnl != -1) && (layer_graph[nnl].layer_type == LAYER_RELU)) {
            layer_graph[l].layer_type = LAYER_HLSINF;
            strcpy(layer_graph[l].sz_layer_name, "HLSinf (pad +conv + relu)");
            layer_graph[l].i = ceil((float)layer_graph[nl].i/CPI) * CPI;
            layer_graph[l].o = ceil((float)layer_graph[nl].o/CPO) * CPO;
            layer_graph[l].ho = layer_graph[nnl].ho;
            layer_graph[l].wo = layer_graph[nnl].wo;
            layer_graph[l].kh = layer_graph[nl].kh;
            layer_graph[l].kw = layer_graph[nl].kw;
            layer_graph[l].sh = layer_graph[nl].sh;
            layer_graph[l].sw = layer_graph[nl].sw;
            layer_graph[l].pt += layer_graph[nl].pt;
            layer_graph[l].pb += layer_graph[nl].pb;
            layer_graph[l].pl += layer_graph[nl].pl;
            layer_graph[l].pr += layer_graph[nl].pr;
            layer_graph[l].conv_layer = layer_graph[nl].conv_layer;
            layer_graph[l].output_tensor = layer_graph[nnl].output_tensor;
            layer_graph[l].weight_tensor = layer_graph[nl].weight_tensor;
            layer_graph[l].bias_tensor = layer_graph[nl].bias_tensor;
            layer_graph[l].apply_conv = 1;
            layer_graph[l].apply_relu = 1;
            layer_graph[l].bn_relu_factor = layer_graph[nnl].relu_factor;
            layer_graph[nl].layer_type = LAYER_NONE;
            layer_graph[nnl].layer_type = LAYER_NONE;
            for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nnl].next[x];
            update_prevs(nl, l);
            update_prevs(nnl, l);
            update_nexts(nl, l);
            update_nexts(nnl, l);
          }
        }
      }
    }
  }
}

void apply_pad_conv_maxp() {}
void apply_pad_conv_leakyrelu() {}

void apply_select_conv_bn_leakyrelu() {
  if (!hlsinf_conv_support) return;
  if (!hlsinf_bn_support) return;
  if (!hlsinf_bn_relu_support) return;
  if (!hlsinf_input_offset_support) return;

  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_SELECT) {
      int nl = layer_graph[l].next[0];
      if ((nl != -1) && (layer_graph[nl].layer_type == LAYER_CONV)) {
        if ((layer_graph[nl].wo <= hlsinf_wo_max) && (layer_graph[nl].kh <= 3) && (layer_graph[nl].kw <= 3)) {
          int nnl = layer_graph[nl].next[0];
          if ((nnl != -1) && (layer_graph[nnl].layer_type == LAYER_BATCH_NORM)) {
            int nnnl = layer_graph[nnl].next[0];
            if ((nnnl != -1) && (layer_graph[nnnl].layer_type == LAYER_LEAKY_RELU)) {
              layer_graph[l].layer_type = LAYER_HLSINF;
              strcpy(layer_graph[l].sz_layer_name, "HLSinf (select + conv + bn + relu)");
   	      layer_graph[l].i = ceil((float)layer_graph[nl].i/CPI) * CPI;
              layer_graph[l].o = ceil((float)layer_graph[nl].o/CPO) * CPO;
              layer_graph[l].ho = layer_graph[nnnl].ho;
              layer_graph[l].wo = layer_graph[nnnl].wo;
	      layer_graph[l].kh = layer_graph[nl].kh;
              layer_graph[l].kw = layer_graph[nl].kw;
              layer_graph[l].sh = layer_graph[nl].sh;
              layer_graph[l].sw = layer_graph[nl].sw;
              layer_graph[l].pt = layer_graph[nl].pt;
              layer_graph[l].pb = layer_graph[nl].pb;
              layer_graph[l].pl = layer_graph[nl].pl;
              layer_graph[l].pr = layer_graph[nl].pr;
              layer_graph[l].conv_layer = layer_graph[nl].conv_layer;
              layer_graph[l].bn_layer = layer_graph[nnl].bn_layer;
      	      layer_graph[l].output_tensor = layer_graph[nnnl].output_tensor;
              layer_graph[l].weight_tensor = layer_graph[nl].weight_tensor;
              layer_graph[l].bias_tensor = layer_graph[nl].bias_tensor;
              layer_graph[l].apply_conv = 1;
    	      layer_graph[l].apply_bn = 1;
              layer_graph[l].apply_bn_relu = 1;
              layer_graph[l].bn_relu_factor = layer_graph[nnnl].relu_factor;
              layer_graph[l].bn_mean_tensor = layer_graph[nnl].bn_mean_tensor;
              layer_graph[l].bn_var_tensor = layer_graph[nnl].bn_var_tensor;
              layer_graph[l].bn_bn_g_tensor = layer_graph[nnl].bn_bn_g_tensor;
              layer_graph[l].bn_bn_b_tensor = layer_graph[nnl].bn_bn_b_tensor;
              layer_graph[nl].layer_type = LAYER_NONE;
              layer_graph[nnl].layer_type = LAYER_NONE;
              layer_graph[nnnl].layer_type = LAYER_NONE;
              for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nnnl].next[x];
              update_prevs(nl, l);
              update_prevs(nnl, l);        
              update_prevs(nnnl, l);        
              update_nexts(nl, l);
              update_nexts(nnl, l);
              update_nexts(nnnl, l);
	    }
          }
        }
      }
    }
  }
}

void apply_select_conv() {
  if (!hlsinf_conv_support) return;
  if (!hlsinf_input_offset_support) return;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_SELECT) {
      int nl = layer_graph[l].next[0];
      if ((nl != -1) && (layer_graph[nl].layer_type == LAYER_CONV)) {
        if ((layer_graph[nl].wo <= hlsinf_wo_max) && (layer_graph[nl].kh <= 3) && (layer_graph[nl].kw <= 3)) {
          layer_graph[l].layer_type = LAYER_HLSINF;
          strcpy(layer_graph[l].sz_layer_name, "HLSinf (select + conv)");	  
          layer_graph[l].i = ceil((float)layer_graph[nl].i/CPI) * CPI;
          layer_graph[l].o = ceil((float)layer_graph[nl].o/CPO) * CPO;
          layer_graph[l].ho = layer_graph[nl].ho;
          layer_graph[l].wo = layer_graph[nl].wo;
	  layer_graph[l].kh = layer_graph[nl].kh;
          layer_graph[l].kw = layer_graph[nl].kw;
          layer_graph[l].sh = layer_graph[nl].sh;
          layer_graph[l].sw = layer_graph[nl].sw;
          layer_graph[l].pt = layer_graph[nl].pt;
          layer_graph[l].pb = layer_graph[nl].pb;
          layer_graph[l].pl = layer_graph[nl].pl;
          layer_graph[l].pr = layer_graph[nl].pr;
          layer_graph[l].conv_layer = layer_graph[nl].conv_layer;
          layer_graph[l].output_tensor = layer_graph[nl].output_tensor;
          layer_graph[l].weight_tensor = layer_graph[nl].weight_tensor;
          layer_graph[l].bias_tensor = layer_graph[nl].bias_tensor;
          layer_graph[l].apply_conv = 1;
          layer_graph[nl].layer_type = LAYER_NONE;
          for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nl].next[x];
          update_prevs(nl, l);
          update_nexts(nl, l);
	}
      }
    }
  }
}


void apply_pad_conv_bn_leakyrelu() {
  if (!hlsinf_conv_support) return;
  if (!hlsinf_bn_support) return;
  if (!hlsinf_bn_relu_support) return;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_PAD) {
      int nl = layer_graph[l].next[0];
      if ((nl != -1) && (layer_graph[nl].layer_type == LAYER_CONV)) {
        if ((layer_graph[nl].wo <= hlsinf_wo_max) && (layer_graph[nl].kh <= 3) && (layer_graph[nl].kw <= 3)) {
	  int nnl = layer_graph[nl].next[0];
	  if ((nnl != -1) && (layer_graph[nnl].layer_type == LAYER_BATCH_NORM)) {
	    int nnnl = layer_graph[nnl].next[0];
	    if ((nnnl != -1) && (layer_graph[nnnl].layer_type == LAYER_LEAKY_RELU)) {
	      layer_graph[l].layer_type = LAYER_HLSINF;
              strcpy(layer_graph[l].sz_layer_name, "HLSinf (pad + conv + bn + lrelu)");
              layer_graph[l].i = ceil((float)layer_graph[nl].i/CPI) * CPI;
              layer_graph[l].o = ceil((float)layer_graph[nl].o/CPO) * CPO;
              layer_graph[l].ho = layer_graph[nnnl].ho;
              layer_graph[l].wo = layer_graph[nnnl].wo;
	      layer_graph[l].kh = layer_graph[nl].kh;
              layer_graph[l].kw = layer_graph[nl].kw;
              layer_graph[l].sh = layer_graph[nl].sh;
              layer_graph[l].sw = layer_graph[nl].sw;
              layer_graph[l].pt += layer_graph[nl].pt;
              layer_graph[l].pb += layer_graph[nl].pb;
              layer_graph[l].pl += layer_graph[nl].pl;
              layer_graph[l].pr += layer_graph[nl].pr;
              layer_graph[l].conv_layer = layer_graph[nl].conv_layer;        
              layer_graph[l].bn_layer = layer_graph[nnl].bn_layer;
              layer_graph[l].output_tensor = layer_graph[nnnl].output_tensor;
	      layer_graph[l].weight_tensor = layer_graph[nl].weight_tensor;
	      layer_graph[l].bias_tensor = layer_graph[nl].bias_tensor;
              layer_graph[l].apply_conv = 1;
    	      layer_graph[l].apply_bn = 1;
	      layer_graph[l].apply_bn_relu = 1;
	      layer_graph[l].bn_relu_factor = layer_graph[nnnl].relu_factor;
              layer_graph[l].bn_mean_tensor = layer_graph[nnl].bn_mean_tensor;
              layer_graph[l].bn_var_tensor = layer_graph[nnl].bn_var_tensor;
              layer_graph[l].bn_bn_g_tensor = layer_graph[nnl].bn_bn_g_tensor;
              layer_graph[l].bn_bn_b_tensor = layer_graph[nnl].bn_bn_b_tensor;
              layer_graph[nl].layer_type = LAYER_NONE;
              layer_graph[nnl].layer_type = LAYER_NONE;
              layer_graph[nnnl].layer_type = LAYER_NONE;
              for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nnnl].next[x];
              update_prevs(nl, l);        
              update_prevs(nnl, l);        
              update_prevs(nnnl, l);       
              update_nexts(nl, l);
              update_nexts(nnl, l);
              update_nexts(nnnl, l);
	    }
	  }
	}
      }
    }
  }
}


void apply_pad_conv_bn() {}
void apply_pad_conv_add() {}

void apply_pad_conv() {
  if (!hlsinf_conv_support) return;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_PAD) {
      int nl = layer_graph[l].next[0];
      if ((nl != -1) && (layer_graph[nl].layer_type == LAYER_CONV)) {
        if ((layer_graph[nl].wo <= hlsinf_wo_max) && (layer_graph[nl].kh <= 3) && (layer_graph[nl].kw <= 3)) {
  	  layer_graph[l].layer_type = LAYER_HLSINF;
          strcpy(layer_graph[l].sz_layer_name, "HLSinf (pad + conv)");
          layer_graph[l].i = ceil((float)layer_graph[nl].i/CPI) * CPI;
          layer_graph[l].o = ceil((float)layer_graph[nl].o/CPO) * CPO;
          layer_graph[l].ho = layer_graph[nl].ho;
          layer_graph[l].wo = layer_graph[nl].wo;
	  layer_graph[l].kh = layer_graph[nl].kh;
          layer_graph[l].kw = layer_graph[nl].kw;
          layer_graph[l].sh = layer_graph[nl].sh;
          layer_graph[l].sw = layer_graph[nl].sw;
	  layer_graph[l].pt += layer_graph[nl].pt;
          layer_graph[l].pb += layer_graph[nl].pb;
          layer_graph[l].pl += layer_graph[nl].pl;
          layer_graph[l].pr += layer_graph[nl].pr;
          layer_graph[l].apply_conv = 1;
          layer_graph[l].conv_layer = layer_graph[nl].conv_layer;        
	  layer_graph[l].output_tensor = layer_graph[nl].output_tensor;
          layer_graph[nl].layer_type = LAYER_NONE;
          for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nl].next[x];
          update_prevs(nl, l);        
          update_nexts(nl, l);
	}
      }
    }
  }
}

void apply_conv_resize() {}

void apply_conv_add_relu() {
  if (!hlsinf_conv_support) return;
  if (!hlsinf_add_support) return;
  if (!hlsinf_add_relu_support) return;

  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONV) {
      if ((layer_graph[l].wo <= hlsinf_wo_max) && (layer_graph[l].kh <= 3) && (layer_graph[l].kw <= 3)) {
        int nl = layer_graph[l].next[0];
        if (layer_graph[nl].layer_type == LAYER_ADD) {
          int nnl = layer_graph[nl].next[0];
	  if ((nnl != -1) && (layer_graph[nnl].layer_type == LAYER_RELU)) {
            layer_graph[l].layer_type = LAYER_HLSINF;
            strcpy(layer_graph[l].sz_layer_name, "HLSinf (conv + add + relu)");
            layer_graph[l].i = ceil((float)layer_graph[l].i/CPI) * CPI;
            layer_graph[l].o = ceil((float)layer_graph[l].o/CPO) * CPO;
            layer_graph[l].apply_conv = 1;
            layer_graph[l].apply_add = 1;
	    layer_graph[l].apply_add_relu = 1;
	    layer_graph[l].relu_factor = layer_graph[nnl].relu_factor;
            layer_graph[l].add_tensor = layer_graph[nl].add_tensor;
            layer_graph[l].output_tensor = layer_graph[nnl].output_tensor;
            layer_graph[nl].layer_type = LAYER_NONE;
            layer_graph[nnl].layer_type = LAYER_NONE;
            for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nnl].next[x];
            update_prevs(nl, l);
	    update_prevs(nnl, l);
            update_nexts(nl, l);
            update_nexts(nnl, l);
	  }
        }
      }
    }
  }
}

void apply_conv_relu_bn_add_upsampling() {
  if (!hlsinf_conv_support) return;
  if (!hlsinf_relu_support) return;
  if (!hlsinf_bn_support)   return;
  if (!hlsinf_add_support)  return;
  if (!hlsinf_upsize_support) return;

  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONV) {
      if ((layer_graph[l].wo <= hlsinf_wo_max) && (layer_graph[l].kh <= 3) && (layer_graph[l].kw <= 3)) {
        int nl = layer_graph[l].next[0];
        if (layer_graph[nl].layer_type == LAYER_RELU) {
          int nnl = layer_graph[nl].next[0];
          if ((nnl != -1) && (layer_graph[nnl].layer_type == LAYER_BATCH_NORM)) {
            int nnnl = layer_graph[nnl].next[0];
            if ((nnnl != -1) && (layer_graph[nnnl].layer_type == LAYER_ADD)) {
              int nnnnl = layer_graph[nnnl].next[0];
	      if ((nnnnl != -1) && (layer_graph[nnnnl].layer_type == LAYER_UPSAMPLING)) {
	        layer_graph[l].layer_type = LAYER_HLSINF; 
       		strcpy(layer_graph[l].sz_layer_name, "HLSinf (conv + relu + bn + add + upsampling)");
                layer_graph[l].i = ceil((float)layer_graph[l].i/CPI) * CPI;
                layer_graph[l].o = ceil((float)layer_graph[l].o/CPO) * CPO;
                layer_graph[l].apply_conv = 1;
                layer_graph[l].apply_relu = 1;
                layer_graph[l].apply_bn   = 1;
                layer_graph[l].apply_add  = 1;
                layer_graph[l].output_tensor = layer_graph[nnnnl].output_tensor;
                layer_graph[l].bn_mean_tensor = layer_graph[nnl].bn_mean_tensor;
                layer_graph[l].bn_var_tensor = layer_graph[nnl].bn_var_tensor;
                layer_graph[l].bn_bn_g_tensor = layer_graph[nnl].bn_bn_g_tensor;
                layer_graph[l].bn_bn_b_tensor = layer_graph[nnl].bn_bn_b_tensor;
                layer_graph[l].bn_layer = layer_graph[nnl].bn_layer;
                layer_graph[l].add_tensor = layer_graph[nnnl].add_tensor;
                layer_graph[l].upscale_factor = layer_graph[nnnnl].upscale_factor;
  	        layer_graph[nl].layer_type = LAYER_NONE;
                layer_graph[nnl].layer_type = LAYER_NONE;
                layer_graph[nnnl].layer_type = LAYER_NONE;
		layer_graph[nnnnl].layer_type = LAYER_NONE;
                for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nnnnl].next[x];
                update_prevs(nl, l);
                update_prevs(nnl, l);
                update_prevs(nnnl, l);
                update_prevs(nnnnl, l);
		update_nexts(nl, l);
                update_nexts(nnl, l);
                update_nexts(nnnl, l);
                update_nexts(nnnnl, l);
	      }
	    }
          }
        }
      }
    }
  }
}

void apply_conv_relu_bn_add() {
  if (!hlsinf_conv_support) return;
  if (!hlsinf_relu_support) return;
  if (!hlsinf_bn_support)   return;
  if (!hlsinf_add_support)  return;

  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONV) {
      if ((layer_graph[l].wo <= hlsinf_wo_max) && (layer_graph[l].kh <= 3) && (layer_graph[l].kw <= 3)) {
        int nl = layer_graph[l].next[0];
        if (layer_graph[nl].layer_type == LAYER_RELU) {
          int nnl = layer_graph[nl].next[0];
          if ((nnl != -1) && (layer_graph[nnl].layer_type == LAYER_BATCH_NORM)) {
            int nnnl = layer_graph[nnl].next[0];
	    if ((nnnl != -1) && (layer_graph[nnnl].layer_type == LAYER_ADD)) {
	      layer_graph[l].layer_type = LAYER_HLSINF;
              strcpy(layer_graph[l].sz_layer_name, "HLSinf (conv + relu + bn + add)");
              layer_graph[l].i = ceil((float)layer_graph[l].i/CPI) * CPI;
              layer_graph[l].o = ceil((float)layer_graph[l].o/CPO) * CPO;
              layer_graph[l].apply_conv = 1;
              layer_graph[l].apply_relu = 1;
              layer_graph[l].apply_bn   = 1;
	      layer_graph[l].apply_add  = 1;
              layer_graph[l].output_tensor = layer_graph[nnnl].output_tensor;
              layer_graph[l].bn_mean_tensor = layer_graph[nnl].bn_mean_tensor;
              layer_graph[l].bn_var_tensor = layer_graph[nnl].bn_var_tensor;
              layer_graph[l].bn_bn_g_tensor = layer_graph[nnl].bn_bn_g_tensor;
              layer_graph[l].bn_bn_b_tensor = layer_graph[nnl].bn_bn_b_tensor;
              layer_graph[l].bn_layer = layer_graph[nnl].bn_layer;
              layer_graph[l].add_tensor = layer_graph[nnnl].add_tensor;
              layer_graph[nl].layer_type = LAYER_NONE;
              layer_graph[nnl].layer_type = LAYER_NONE;
	      layer_graph[nnnl].layer_type = LAYER_NONE;
              for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nnnl].next[x];
              update_prevs(nl, l);
              update_prevs(nnl, l);
	      update_prevs(nnnl, l);
              update_nexts(nl, l);
              update_nexts(nnl, l);
	      update_nexts(nnnl, l);
	    }
          }
        }
      }
    }
  }
}

void apply_conv_relu_bn() {
  if (!hlsinf_conv_support) return;
  if (!hlsinf_relu_support) return;
  if (!hlsinf_bn_support)   return;

  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONV) {
      if ((layer_graph[l].wo <= hlsinf_wo_max) && (layer_graph[l].kh <= 3) && (layer_graph[l].kw <= 3)) {
        int nl = layer_graph[l].next[0];
        if (layer_graph[nl].layer_type == LAYER_RELU) {
          int nnl = layer_graph[nl].next[0];
          if ((nnl != -1) && (layer_graph[nnl].layer_type == LAYER_BATCH_NORM)) {
	    layer_graph[l].layer_type = LAYER_HLSINF;
            strcpy(layer_graph[l].sz_layer_name, "HLSinf (conv + relu + bn)");
            layer_graph[l].i = ceil((float)layer_graph[l].i/CPI) * CPI;
            layer_graph[l].o = ceil((float)layer_graph[l].o/CPO) * CPO;
            layer_graph[l].apply_conv = 1;
            layer_graph[l].apply_relu = 1;
	    layer_graph[l].apply_bn   = 1;
            layer_graph[l].output_tensor = layer_graph[nnl].output_tensor;
            layer_graph[l].bn_mean_tensor = layer_graph[nnl].bn_mean_tensor;
            layer_graph[l].bn_var_tensor = layer_graph[nnl].bn_var_tensor;
            layer_graph[l].bn_bn_g_tensor = layer_graph[nnl].bn_bn_g_tensor;
            layer_graph[l].bn_bn_b_tensor = layer_graph[nnl].bn_bn_b_tensor;
            layer_graph[l].bn_layer = layer_graph[nnl].bn_layer;
            layer_graph[nl].layer_type = LAYER_NONE;
	    layer_graph[nnl].layer_type = LAYER_NONE;
            for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nnl].next[x];
            update_prevs(nl, l);
	    update_prevs(nnl, l);
            update_nexts(nl, l);
	    update_nexts(nnl, l);
	  }
        }
      }
    }
  }
}

void apply_conv_relu_maxp_resize() {}
void apply_conv_relu_maxp_bn() {}

void apply_conv_relu_maxp() {
  if (!hlsinf_conv_support) return;
  if (!hlsinf_maxp_support) return;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONV) {
      if ((layer_graph[l].wo <= hlsinf_wo_max) && (layer_graph[l].kh <= 3) && (layer_graph[l].kw <= 3)) {
        int nl = layer_graph[l].next[0];
	if (layer_graph[nl].layer_type == LAYER_RELU) {
	  int nnl = layer_graph[nl].next[0];
          if (layer_graph[nnl].layer_type == LAYER_MAX_POOL) {
            layer_graph[l].layer_type = LAYER_HLSINF;
            strcpy(layer_graph[l].sz_layer_name, "HLSinf (conv + relu + maxp)");
            layer_graph[l].i = ceil((float)layer_graph[l].i/CPI) * CPI;
            layer_graph[l].o = ceil((float)layer_graph[l].o/CPO) * CPO;
            layer_graph[l].ho = layer_graph[nnl].ho;
            layer_graph[l].wo = layer_graph[nnl].wo;
            layer_graph[l].apply_conv = 1;
            layer_graph[l].apply_maxp = 1;
	    layer_graph[l].apply_relu = 1;
	    layer_graph[l].relu_factor = layer_graph[nl].relu_factor;
            layer_graph[l].output_tensor = layer_graph[nnl].output_tensor;
            layer_graph[nl].layer_type = LAYER_NONE;
            layer_graph[nnl].layer_type = LAYER_NONE;
            for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nnl].next[x];
            update_prevs(nl, l);
	    update_prevs(nnl, l);
            update_nexts(nl, l);
            update_nexts(nnl, l);
	  }
        }
      }
    }
  }
}

void apply_conv_relu_resize() {}
void apply_conv_leakyrelu() {}

void apply_conv_bn() {
  if (!hlsinf_conv_support) return;
  if (!hlsinf_bn_support) return;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONV) {
      if ((layer_graph[l].wo <= hlsinf_wo_max) && (layer_graph[l].kh <= 3) && (layer_graph[l].kw <= 3)) {
        int nl = layer_graph[l].next[0];
        if (layer_graph[nl].layer_type == LAYER_BATCH_NORM) {
          layer_graph[l].layer_type = LAYER_HLSINF;
          strcpy(layer_graph[l].sz_layer_name, "HLSinf (conv + bn)");
          layer_graph[l].i = ceil((float)layer_graph[l].i/CPI) * CPI;
          layer_graph[l].o = ceil((float)layer_graph[l].o/CPO) * CPO;
          layer_graph[l].ho = layer_graph[nl].ho;
          layer_graph[l].wo = layer_graph[nl].wo;
	  layer_graph[l].apply_conv = 1;
	  layer_graph[l].apply_bn = 1;
          layer_graph[l].output_tensor = layer_graph[nl].output_tensor;
	  layer_graph[l].bn_mean_tensor = layer_graph[nl].bn_mean_tensor;
          layer_graph[l].bn_var_tensor = layer_graph[nl].bn_var_tensor;
          layer_graph[l].bn_bn_g_tensor = layer_graph[nl].bn_bn_g_tensor;
          layer_graph[l].bn_bn_b_tensor = layer_graph[nl].bn_bn_b_tensor;
          layer_graph[l].bn_layer = layer_graph[nl].bn_layer;
          layer_graph[nl].layer_type = LAYER_NONE;
          for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nl].next[x];
          update_prevs(nl, l);        
          update_nexts(nl, l);
	}
      }
    }
  }
}

void apply_conv() {
  if (!hlsinf_conv_support) return;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONV) {
      if ((layer_graph[l].wo <= hlsinf_wo_max) && (layer_graph[l].kh <= 3) && (layer_graph[l].kw <= 3)) {
        layer_graph[l].layer_type = LAYER_HLSINF;
        strcpy(layer_graph[l].sz_layer_name, "HLSinf (conv)");
        layer_graph[l].i = ceil((float)layer_graph[l].i/CPI) * CPI;
        layer_graph[l].o = ceil((float)layer_graph[l].o/CPO) * CPO;
        layer_graph[l].apply_conv = 1;
      }
    }
  }
}

void apply_maxp() {
  if (!hlsinf_maxp_support) return;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_MAX_POOL) {
      layer_graph[l].layer_type = LAYER_HLSINF;
      strcpy(layer_graph[l].sz_layer_name, "HLSinf (maxp)");
      layer_graph[l].i = ceil((float)layer_graph[l].i/CPI) * CPI;
      layer_graph[l].o = ceil((float)layer_graph[l].o/CPO) * CPO;
      layer_graph[l].kh = 3;  // we configure conv parameters so we have a transparent convolution
      layer_graph[l].kw = 3;
      layer_graph[l].sh = 1;
      layer_graph[l].sw = 1;
      layer_graph[l].pt = 0;
      layer_graph[l].pb = 2;
      layer_graph[l].pl = 0;
      layer_graph[l].pr = 2;
      layer_graph[l].apply_maxp = 1;
    }
  }
}

void apply_upsampling() {
  if (!hlsinf_upsize_support) return;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_UPSAMPLING) {
      layer_graph[l].layer_type = LAYER_HLSINF;
      strcpy(layer_graph[l].sz_layer_name, "HLSinf (upsampling)");
      layer_graph[l].i = ceil((float)layer_graph[l].i/CPI) * CPI;
      layer_graph[l].o = ceil((float)layer_graph[l].o/CPO) * CPO;
    }
  }
}

void apply_conv_bn_leakyrelu_upsampling() {
  if (!hlsinf_conv_support) return;
  if (!hlsinf_bn_support) return;
  if (!hlsinf_bn_relu_support) return;
  if (!hlsinf_upsize_support) return;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONV) {
      if ((layer_graph[l].wo <= hlsinf_wo_max) && (layer_graph[l].kh <= 3) && (layer_graph[l].kw <= 3)) {
        int nl = layer_graph[l].next[0];
        if ((nl != -1) && (layer_graph[nl].layer_type == LAYER_BATCH_NORM)) {
          int nnl = layer_graph[nl].next[0];
          if ((nnl != -1) && (layer_graph[nnl].layer_type == LAYER_LEAKY_RELU)) {
	    int nnnl = layer_graph[nnl].next[0];
	    if ((nnnl != -1) && (layer_graph[nnnl].layer_type == LAYER_UPSAMPLING)) {
              layer_graph[l].layer_type = LAYER_HLSINF;
              strcpy(layer_graph[l].sz_layer_name, "HLSinf (conv + bn + lrelu + upsampling)");
              layer_graph[l].i = ceil((float)layer_graph[l].i/CPI) * CPI;
              layer_graph[l].o = ceil((float)layer_graph[l].o/CPO) * CPO;
              layer_graph[l].ho = layer_graph[nnnl].ho;
              layer_graph[l].wo = layer_graph[nnnl].wo;
	      layer_graph[l].apply_conv = 1;
              layer_graph[l].apply_bn = 1;
              layer_graph[l].apply_bn_relu = 1;
              layer_graph[l].bn_layer = layer_graph[nl].bn_layer;
              layer_graph[l].output_tensor = layer_graph[nnl].output_tensor;
              layer_graph[l].bn_mean_tensor = layer_graph[nl].bn_mean_tensor;
              layer_graph[l].bn_var_tensor = layer_graph[nl].bn_var_tensor;
              layer_graph[l].bn_bn_g_tensor = layer_graph[nl].bn_bn_g_tensor;
              layer_graph[l].bn_bn_b_tensor = layer_graph[nl].bn_bn_b_tensor;
              layer_graph[l].bn_relu_factor = layer_graph[nnl].relu_factor;
	      layer_graph[l].upscale_factor = layer_graph[nnnl].upscale_factor;
              layer_graph[nl].layer_type = LAYER_NONE;
              layer_graph[nnl].layer_type = LAYER_NONE;
              layer_graph[nnnl].layer_type = LAYER_NONE;
              for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nnnl].next[x];
              update_prevs(nl, l);        
              update_prevs(nnl, l);        
              update_prevs(nnnl, l);       
              update_nexts(nl, l);
              update_nexts(nnl, l);
              update_nexts(nnnl, l);
	    }
	  }
        }
      }
    }
  }
}

void apply_conv_bn_leakyrelu() {
  if (!hlsinf_conv_support) return;
  if (!hlsinf_bn_support) return;
  if (!hlsinf_bn_relu_support) return;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONV) {
      if ((layer_graph[l].wo <= hlsinf_wo_max) && (layer_graph[l].kh <= 3) && (layer_graph[l].kw <= 3)) {
        int nl = layer_graph[l].next[0];
        if ((nl != -1) && (layer_graph[nl].layer_type == LAYER_BATCH_NORM)) {
	  int nnl = layer_graph[nl].next[0];
	  if ((nnl != -1) && (layer_graph[nnl].layer_type == LAYER_LEAKY_RELU)) {
            layer_graph[l].layer_type = LAYER_HLSINF;
            strcpy(layer_graph[l].sz_layer_name, "HLSinf (conv + bn + lrelu)");
  	    layer_graph[l].i = ceil((float)layer_graph[l].i/CPI) * CPI;
	    layer_graph[l].o = ceil((float)layer_graph[l].o/CPO) * CPO;
            layer_graph[l].ho = layer_graph[nnl].ho;
            layer_graph[l].wo = layer_graph[nnl].wo;
  	    layer_graph[l].apply_conv = 1;
  	    layer_graph[l].apply_bn = 1;
	    layer_graph[l].apply_bn_relu = 1;
            layer_graph[l].output_tensor = layer_graph[nnl].output_tensor;
            layer_graph[l].bn_mean_tensor = layer_graph[nl].bn_mean_tensor;
            layer_graph[l].bn_var_tensor = layer_graph[nl].bn_var_tensor;
            layer_graph[l].bn_bn_g_tensor = layer_graph[nl].bn_bn_g_tensor;
            layer_graph[l].bn_bn_b_tensor = layer_graph[nl].bn_bn_b_tensor;
	    layer_graph[l].bn_relu_factor = layer_graph[nnl].relu_factor;
            layer_graph[l].bn_layer = layer_graph[nl].bn_layer;
            layer_graph[nl].layer_type = LAYER_NONE;
	    layer_graph[nnl].layer_type = LAYER_NONE;
            for (int x=0; x<MAX_CHILDS; x++) layer_graph[l].next[x] = layer_graph[nnl].next[x];
            update_prevs(nl, l);        
            update_prevs(nnl, l);        
            update_nexts(nl, l);
            update_nexts(nnl, l);
	  }
	}
      }
    }
  }
}

void apply_hlsinf_layers() {
  apply_select_conv_bn_leakyrelu();	
  apply_select_conv();
  apply_pad_conv_sigmoid_tanh_maxp_add();
  apply_pad_conv_sigmoid_tanh_maxp();
  apply_pad_conv_relu_maxp();
  apply_pad_conv_relu();
  apply_pad_conv_maxp();
  apply_pad_conv_leakyrelu();
  apply_pad_conv_bn_leakyrelu();
  apply_pad_conv_bn();
  apply_pad_conv_add();
  apply_pad_conv();
  apply_conv_mult_clamp_relu_maxp();
  apply_conv_mult_clamp_relu();
  apply_conv_mult();
  apply_conv_div_clamp_relu_maxp();
  apply_conv_div_clamp_relu();
  apply_conv_div();
  apply_conv_softplus_tanh_mult_add();
  apply_conv_softplus_tanh_mult();
  apply_conv_resize();
  apply_conv_add_relu();
  apply_conv_relu_bn_add_upsampling();
  apply_conv_relu_bn_add();
  apply_conv_relu_bn();
  apply_conv_relu_maxp_resize();
  apply_conv_relu_maxp_bn();
  apply_conv_relu_maxp();
  apply_conv_relu_resize();
  apply_conv_relu();
  apply_conv_leakyrelu();
  apply_conv_maxp();
  apply_conv_add();
  apply_conv_bn_leakyrelu_upsampling();
  apply_conv_bn_leakyrelu();
  apply_conv_bn();
  apply_conv();
  apply_maxp();
  //apply_upsampling();
}

int remove_leftover_layers() {
  int nl = 0;
  for (int l=0; l<graph_size; l++) {
    int any_child = 0;
    for (int p=0; p<3; p++) if (layer_graph[l].next[p] != -1) any_child = 1;
    if ((any_child==0) && (layer_graph[l].out_layer == 0)) {
      nl++;
      layer_graph[l].layer_type = LAYER_NONE;
      for (int x=0; x<graph_size; x++) {
	for (int p=0; p<MAX_CHILDS; p++) {
          if (layer_graph[x].next[p] == l) layer_graph[x].next[p] = -1;
	}
      }
    }
  }
  return nl;
}
	

int is_cpu_layer(int l) {
  if (layer_graph[l].layer_type == LAYER_HLSINF) return 0;
  if (layer_graph[l].layer_type == LAYER_DUMMY_CONCAT) return 0;
  return 1;
}

int is_fpga_layer(int l) {
  return !is_cpu_layer(l);
}

void add_transform_layers() {

  int next_layer = graph_size;
  int next_buffer = num_buffers;

  // we sweep the graph in search of CPU->FPGA and FPGA->CPU transitions
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type != LAYER_NONE) {
      if (is_cpu_layer(l)) {
        // let's see if there is at least one next layer with cpu->fpga
        int found = 0;
        for (int p=0; p<MAX_CHILDS; p++) if ((layer_graph[l].next[p] != -1) && (is_fpga_layer(layer_graph[l].next[p]))) found = 1;
        if (found) {
	  // transform layer
          layer_graph[next_layer].layer_type = LAYER_TRANSFORM;
	  layer_graph[next_layer].i = layer_graph[l].o;
	  layer_graph[next_layer].o = layer_graph[l].o;
	  layer_graph[next_layer].hi = layer_graph[l].ho;
	  layer_graph[next_layer].wi = layer_graph[l].wo;
	  layer_graph[next_layer].ho = layer_graph[l].ho;
	  layer_graph[next_layer].wo = layer_graph[l].wo;
          layer_graph[next_layer].layer = nullptr;
          layer_graph[next_layer].input_tensor = nullptr;
          layer_graph[next_layer].output_tensor = nullptr;
          layer_graph[next_layer].weight_tensor = nullptr;
          layer_graph[next_layer].bias_tensor = nullptr;
          layer_graph[next_layer].add_tensor = nullptr;
          layer_graph[next_layer].bn_mean_tensor = nullptr;
          layer_graph[next_layer].bn_var_tensor = nullptr;
          layer_graph[next_layer].bn_bn_g_tensor = nullptr;
          layer_graph[next_layer].bn_bn_b_tensor = nullptr;
	  layer_graph[next_layer].cpu2fpga = 1;
	  layer_graph[next_layer].fpga2cpu = 0;
	  layer_graph[next_layer].transform = 1;
	  // buffers
	  layer_graph[next_layer].input_buffer[0] = layer_graph[l].output_buffer;
	  layer_graph[next_layer].input_buffer[1] = -1;
	  layer_graph[next_layer].output_buffer = next_buffer;
	  // replace output buffer of layer l by next buffer in all the graph (except in layer l and in cpu layers)) 
	  int old_buff = layer_graph[l].output_buffer;
	  for (int x=0; x<graph_size; x++) {
	    if (x != l) {
              if (is_fpga_layer(x) && (layer_graph[x].input_buffer[0] == old_buff)) layer_graph[x].input_buffer[0] = next_buffer;
              if (is_fpga_layer(x) && (layer_graph[x].input_buffer[1] == old_buff)) layer_graph[x].input_buffer[1] = next_buffer;
//              if (layer_graph[x].output_buffer == old_buff) layer_graph[x].output_buffer = next_buffer;
	    }
	  }
	  next_buffer++;
	  // the transform layer points to all fpga layers following the current layer
	  for (int p=0; p<MAX_CHILDS; p++) {
            if ((layer_graph[l].next[p] != -1) && (is_fpga_layer(layer_graph[l].next[p]))) {
	      layer_graph[next_layer].next[p] = layer_graph[l].next[p];
   	    } else { 
	      layer_graph[next_layer].next[p] = -1;
	    }
	  }
	  // Only one link between the current layer ant the transform layer
	  int already_linked = 0;
	  for (int p=0; p<MAX_CHILDS; p++) {
	    if ((layer_graph[l].next[p] != -1) && (is_fpga_layer(layer_graph[l].next[p]))) {
	      if (already_linked) layer_graph[l].next[p] = -1;
	      else {
		layer_graph[l].next[p] = next_layer;
		already_linked = 1;
              }
	    }
	  }
	  // Let's fix prev fields
	  layer_graph[next_layer].prev[0] = l; layer_graph[next_layer].prev[1] = -1; layer_graph[next_layer].prev[2] = -1;
	  for (int p=0; p<MAX_CHILDS; p++) {
  	    int x = layer_graph[next_layer].next[p];
	    if (layer_graph[x].prev[0] == l) layer_graph[x].prev[0] = next_layer;
            if (layer_graph[x].prev[1] == l) layer_graph[x].prev[1] = next_layer;
	    if (layer_graph[x].prev[2] == l) layer_graph[x].prev[2] = next_layer;
	  }

	  next_layer++;
        }
      }
      if (is_fpga_layer(l)) {
        // let's see if there is at least one next layer with fpga->cpu
        int found = 0;
        for (int p=0; p<MAX_CHILDS; p++) if ((layer_graph[l].next[p] != -1) && (is_cpu_layer(layer_graph[l].next[p]))) found = 1;
        if (found) {
          layer_graph[next_layer].layer_type = LAYER_TRANSFORM;
          layer_graph[next_layer].i = layer_graph[l].o;
          layer_graph[next_layer].o = layer_graph[l].o;
          layer_graph[next_layer].hi = layer_graph[l].ho;
          layer_graph[next_layer].wi = layer_graph[l].wo;
          layer_graph[next_layer].ho = layer_graph[l].ho;
          layer_graph[next_layer].wo = layer_graph[l].wo;
          layer_graph[next_layer].layer = nullptr;
          layer_graph[next_layer].input_tensor = nullptr;
          layer_graph[next_layer].output_tensor = nullptr;
          layer_graph[next_layer].weight_tensor = nullptr;
          layer_graph[next_layer].bias_tensor = nullptr;
          layer_graph[next_layer].add_tensor = nullptr;
          layer_graph[next_layer].bn_mean_tensor = nullptr;
          layer_graph[next_layer].bn_var_tensor = nullptr;
          layer_graph[next_layer].bn_bn_g_tensor = nullptr;
          layer_graph[next_layer].bn_bn_b_tensor = nullptr;
          layer_graph[next_layer].cpu2fpga = 0;
          layer_graph[next_layer].fpga2cpu = 1;
          layer_graph[next_layer].transform = 1;
          // buffers
          layer_graph[next_layer].input_buffer[0] = layer_graph[l].output_buffer;
          layer_graph[next_layer].input_buffer[1] = -1;
          layer_graph[next_layer].output_buffer = next_buffer;
          // replace output buffer of layer l by next buffer in all the graph (except in layer l)
          int old_buff = layer_graph[l].output_buffer;
          for (int x=0; x<graph_size; x++) {
	    if (x != l) {
              if (is_cpu_layer(x) && (layer_graph[x].input_buffer[0] == old_buff)) layer_graph[x].input_buffer[0] = next_buffer;
              if (is_cpu_layer(x) && (layer_graph[x].input_buffer[1] == old_buff)) layer_graph[x].input_buffer[1] = next_buffer;
//              if (layer_graph[x].output_buffer == old_buff) layer_graph[x].output_buffer = next_buffer;
            }
	  }
          next_buffer++;
          // the transform layer points to all fpga layers following the current layer
          for (int p=0; p<MAX_CHILDS; p++) {
            if ((layer_graph[l].next[p] != -1) && (is_cpu_layer(layer_graph[l].next[p]))) {
	      layer_graph[next_layer].next[p] = layer_graph[l].next[p]; 
	    } else {
	      layer_graph[next_layer].next[p] = -1;
	    }
	  }
          // Only one link between the current layer ant the transform layer
          int already_linked = 0; 
          for (int p=0; p<MAX_CHILDS; p++) {
            if ((layer_graph[l].next[p] != -1) && (is_cpu_layer(layer_graph[l].next[p]))) {
              if (already_linked) layer_graph[l].next[p] = -1;
              else {
                layer_graph[l].next[p] = next_layer;
                already_linked = 1;
              }
            }
          }
          // Let's fix prev fields
          layer_graph[next_layer].prev[0] = l; layer_graph[next_layer].prev[1] = -1; layer_graph[next_layer].prev[2] = -1;
	  for (int p=0; p<MAX_CHILDS; p++) {
            int x = layer_graph[next_layer].next[p];
            if (layer_graph[x].prev[0] == l) layer_graph[x].prev[0] = next_layer;
            if (layer_graph[x].prev[1] == l) layer_graph[x].prev[1] = next_layer;
	    if (layer_graph[x].prev[2] == l) layer_graph[x].prev[2] = next_layer;
	  }

          next_layer++;
        }
      }
    }
  }

  // an output layer must be in cpu, so we add a transform if needed
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type != LAYER_NONE) {
      if (layer_graph[l].out_layer && is_fpga_layer(l)) {
        layer_graph[next_layer].layer_type = LAYER_TRANSFORM;
        layer_graph[next_layer].i = layer_graph[l].o;
        layer_graph[next_layer].o = layer_graph[l].o;
        layer_graph[next_layer].hi = layer_graph[l].ho;
        layer_graph[next_layer].wi = layer_graph[l].wo;
        layer_graph[next_layer].ho = layer_graph[l].ho;
        layer_graph[next_layer].wo = layer_graph[l].wo;
	layer_graph[next_layer].layer = nullptr;
        layer_graph[next_layer].input_tensor = nullptr;
        layer_graph[next_layer].output_tensor = nullptr;
        layer_graph[next_layer].weight_tensor = nullptr;
        layer_graph[next_layer].bias_tensor = nullptr;
        layer_graph[next_layer].add_tensor = nullptr;
        layer_graph[next_layer].bn_mean_tensor = nullptr;
        layer_graph[next_layer].bn_var_tensor = nullptr;
        layer_graph[next_layer].bn_bn_g_tensor = nullptr;
        layer_graph[next_layer].bn_bn_b_tensor = nullptr;
        layer_graph[next_layer].cpu2fpga = 0;
        layer_graph[next_layer].fpga2cpu = 1;
        layer_graph[next_layer].transform = 1;
        // buffers
        layer_graph[next_layer].input_buffer[0] = layer_graph[l].output_buffer;
        layer_graph[next_layer].input_buffer[1] = -1;
        layer_graph[next_layer].output_buffer = next_buffer;
        // replace output buffer of layer l by next buffer in all the graph (except in layer l)
        int old_buff = layer_graph[l].output_buffer;
        for (int x=0; x<graph_size; x++) {
	  if (x != l) {
            if (is_cpu_layer(x) && (layer_graph[x].input_buffer[0] == old_buff)) layer_graph[x].input_buffer[0] = next_buffer;
            if (is_cpu_layer(x) && (layer_graph[x].input_buffer[1] == old_buff)) layer_graph[x].input_buffer[1] = next_buffer;
//            if (layer_graph[x].output_buffer == old_buff) layer_graph[x].output_buffer = next_buffer;
          }
	}
        next_buffer++;
        for (int p=0; p<MAX_CHILDS; p++) layer_graph[next_layer].next[p] = -1;
	layer_graph[next_layer].out_layer = 1;
	layer_graph[l].next[0] = next_layer;
	layer_graph[l].out_layer = 0;
        // Let's fix prev fields
        layer_graph[next_layer].prev[0] = l; layer_graph[next_layer].prev[1] = -1; layer_graph[next_layer].prev[2] = -1;
	for (int p=0; p<MAX_CHILDS; p++) {
          int x = layer_graph[next_layer].next[p];
          if (layer_graph[x].prev[0] == l) layer_graph[x].prev[0] = next_layer;
          if (layer_graph[x].prev[1] == l) layer_graph[x].prev[1] = next_layer;
	  if (layer_graph[x].prev[2] == l) layer_graph[x].prev[2] = next_layer;
	}
	next_layer++;
      }
    }
  }

  graph_size = next_layer;
  num_buffers = next_buffer;
}

int get_num_concat_childs_layer(int l) {
 int num_childs = 0;
 for (int p=0; p<MAX_CHILDS; p++) 
   if ((layer_graph[l].next[p] != -1) && (layer_graph[layer_graph[l].next[p]].layer_type == LAYER_CONCAT)) num_childs++;  
 return num_childs;
}

// remove_concat_layers(). This functions searches the network graph and looks for concat layers that could potentially be
// avoided by rearranging buffers. There are specific combinations of concat layers that may impede their removal. This function
// supports the following combination:
//
//      b0 -> HLSinf -> b1 -> Concat -> b2 -> HLSinf
//      b3 -> HLSinf -> b4 ----^
//
// This combination is transformed into:
//
//      b0 -> HLSinf -> b2_fh -> HLSinf
//      b3 -> HLSinf -> b2_sh
//      
// Another combination of two linked concat layers is this:
//
//                       |--> Concat1 -> b5
//      b0 -> HLSinf -> b1 -> Concat0 -> b2
//      b3 -> HLSinf -> b4 -----^
// 
// This means an HLSinf layer feeding two concat layers. Currently we do not support this.
// Therefore, first combination must prevent the appearance of the later one.
// We simply avoid removing concat layers with parents with more than one child
//
void remove_concat_layers() {

  // we search HLS_a  ---> buffer_a ----> Concat_c ----> buffer_c -----> any_d ----> buffer_d
  //           HLS_b  ---> buffer_b ---------^
  //
  // and replaced with HLS_a ----> buffer_a ----> any_d ----> buffer_d
  //                   HLS_b ----> buffer_b -------^
  //
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONCAT) {
      int parent0 = layer_graph[l].prev[0];
      int parent1 = layer_graph[l].prev[1];
      int child0  = layer_graph[l].next[0];
      if ((parent0 != -1) && (layer_graph[parent0].layer_type == LAYER_HLSINF)) {
	if ((parent1 != -1) && (layer_graph[parent1].layer_type == LAYER_HLSINF)) {
          if (child0 != -1) {
	    printf("found: concat %d parent0 %d parent1 %d child %d\n", l, parent0, parent1, child0);
	    // removal of concat layer
	    layer_graph[l].layer_type = LAYER_NONE;
            // link between parents and child
	    for (int p=0; p<MAX_CHILDS; p++) if (layer_graph[parent0].next[p] == l) layer_graph[parent0].next[p] = child0;
            for (int p=0; p<MAX_CHILDS; p++) if (layer_graph[parent1].next[p] == l) layer_graph[parent1].next[p] = child0;
            layer_graph[child0].prev[0] = parent0;
	    layer_graph[child0].prev[1] = parent1;
	    layer_graph[child0].input_buffer[0] = layer_graph[l].input_buffer[0];
	    layer_graph[child0].input_buffer[1] = layer_graph[l].input_buffer[1];
          }
	}
      }
    }
  }
/*
  // we search for a concat layer within the graph
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONCAT) {
      printf("found %d\n", l);
      // the concat layer has two parents, let's find them (parent0 and parent1)
      int num_found = 0;
      int parent0 = layer_graph[l].prev[0];
      int parent1 = layer_graph[l].prev[1];
      if ((parent0 == -1) || (parent1 == -1)) {printf("Error, did not found two parents for a concat layer\n"); exit(1);}
      
      // parents must have one child as concat
      int num_childs_parent0 = get_num_concat_childs_layer(parent0);
      int num_childs_parent1 = get_num_concat_childs_layer(parent1);

      // The two parents must be HLSinf (FPGA) layers and they need to have only one child in order to proceeed with the removal
      if (is_fpga_layer(parent0) && is_fpga_layer(parent1) && (num_childs_parent0 == 1) && (num_childs_parent1 == 1)) {

        // We do have a combination to eliminate the concat layer, let's proceed

        // removal of concat layer
	layer_graph[l].layer_type = LAYER_NONE;

	// we replace parent0 output buffer by concat output buffer in all the graph, second half of the buffer
	int buf = layer_graph[parent0].output_buffer;
	for (int x=0; x<graph_size; x++) {
	  if (layer_graph[x].layer_type != LAYER_NONE) {
            if (layer_graph[x].input_buffer[0] == buf) {
	      layer_graph[x].input_buffer[0] = layer_graph[l].output_buffer;
	      layer_graph[x].input_offset = (layer_graph[l].o * layer_graph[l].ho * layer_graph[l].wo) / 2;
	    }
            if (layer_graph[x].input_buffer[1] == buf) {
              layer_graph[x].input_buffer[1] = layer_graph[l].output_buffer;
              layer_graph[x].input_offset = (layer_graph[l].o * layer_graph[l].ho * layer_graph[l].wo) / 2;
            }
	    if (layer_graph[x].output_buffer == buf) {
	      layer_graph[x].output_buffer = layer_graph[l].output_buffer;
	      layer_graph[x].output_offset = (layer_graph[l].o * layer_graph[l].ho * layer_graph[l].wo) / 2;
            }
	  }
	}
	// we replace parent1 output buffer by concat output buffer in all the graph, first half of the buffer
        buf = layer_graph[parent1].output_buffer;
        for (int x=0; x<graph_size; x++) {
          if (layer_graph[x].layer_type != LAYER_NONE) {
            if (layer_graph[x].input_buffer[0] == buf) {
              layer_graph[x].input_buffer[0] = layer_graph[l].output_buffer;
              layer_graph[x].input_offset = 0;
            }
            if (layer_graph[x].input_buffer[1] == buf) {
              layer_graph[x].input_buffer[1] = layer_graph[l].output_buffer;
              layer_graph[x].input_offset = 0;
            }
            if (layer_graph[x].output_buffer == buf) {
              layer_graph[x].output_buffer = layer_graph[l].output_buffer;
              layer_graph[x].output_offset = 0;
            }
          }
        }
        // finaly we link the layers correctly, the two parent layers now point to the concat's child layer	
	int next = layer_graph[l].next[0];
	if ((layer_graph[l].next[1] != -1) || (layer_graph[l].next[2] != -1)) {printf("Concat with next in slots 1 or 2 not supported\n"); exit(1);}
	for (int p=0; p<MAX_CHILDS; p++) if (layer_graph[parent0].next[p] == l) layer_graph[parent0].next[p] = next;
        for (int p=0; p<MAX_CHILDS; p++) if (layer_graph[parent1].next[p] == l) layer_graph[parent1].next[p] = next;
	// the next layers after concat now need to have as parents the prev layers of the concat layer
	layer_graph[next].prev[0] = parent0;
	layer_graph[next].prev[1] = parent1;
      }		      
    }
  }*/
}

int get_input_buffer_size_layer(int l) {
  return layer_graph[l].i * layer_graph[l].hi * layer_graph[l].wi;
}

int get_output_buffer_size_layer(int l) {
  return layer_graph[l].o * layer_graph[l].ho * layer_graph[l].wo;
}

void print_buffers_info() {
  for (int b=0; b<num_buffers; b++) {
    printf("buffer %d: size %d, read layers (offset): ", b, buffers[b].size);
    for (int l=0; l<MAX_READ_LAYERS; l++) printf("%3d (%6d) ", buffers[b].read_layers[l], buffers[b].read_offset[l]);
    printf("; write layers (offset): ");
    for (int l=0; l<MAX_WRITE_LAYERS; l++) printf("%3d (%6d) ", buffers[b].write_layers[l], buffers[b].write_offset[l]);
    printf("\n");
  }
}

void add_fpga_buffers() {
  int next_buffer = 0;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type != LAYER_NONE) {
      // input buffer
      if (layer_graph[l].in_layer) {
        layer_graph[l].input_buffer[0] = next_buffer;	      
	buffers[next_buffer].size = get_input_buffer_size_layer(l);
	buffers[next_buffer].read_layers[0] = l;
	buffers[next_buffer].read_offset[0] = layer_graph[l].input_offset;
	for (int x=1; x<MAX_READ_LAYERS; x++) buffers[next_buffer].read_layers[x] = -1;
	for (int x=0; x<MAX_WRITE_LAYERS; x++) buffers[next_buffer].write_layers[x] = -1;
	next_buffer++;
      }
      // output buffer
      if (layer_graph[l].output_buffer == -1) {
        layer_graph[l].output_buffer = next_buffer;
	buffers[next_buffer].size = get_output_buffer_size_layer(l);
	buffers[next_buffer].write_layers[0] = l;
	buffers[next_buffer].write_offset[0] = layer_graph[l].output_offset;
        for (int x=1; x<MAX_WRITE_LAYERS; x++) buffers[next_buffer].write_layers[x] = -1;
        for (int x=0; x<MAX_READ_LAYERS; x++) buffers[next_buffer].read_layers[x] = -1;
	for (int p=0; p<MAX_CHILDS; p++) {
          if (layer_graph[l].next[p] != -1) {
	    int next = layer_graph[l].next[p];
	    if (layer_graph[next].layer_type != LAYER_NONE) {
	      if (layer_graph[next].input_buffer[0] == -1) {
		layer_graph[next].input_buffer[0] = next_buffer; 
              } else {
		layer_graph[next].input_buffer[1] = next_buffer;
	      }
              for (int x=0; x<MAX_READ_LAYERS; x++) if (buffers[next_buffer].read_layers[x] == -1) {buffers[next_buffer].read_layers[x] = next; break;}
	    }
	  }
	}
	next_buffer++;
      }
    }
  }       
  num_buffers = next_buffer;
}

int get_closest_parent(int l) {
  for (int x=0; x<MAX_CHILDS; x++) if (layer_graph[l].next[x] != -1) return layer_graph[l].next[x];
}

void unify_buffers() {
  int next_buffer = num_buffers;

  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type != LAYER_NONE) {
      if (layer_graph[l].layer_type == LAYER_CONCAT) {
	if ((layer_graph[l].prev[0] != -1) && (layer_graph[l].prev[1] != -1)) {

          int parent0 = layer_graph[l].prev[0];
	  int parent1 = layer_graph[l].prev[1];
	  int old_buffer0 = layer_graph[parent0].output_buffer;
	  int old_buffer1 = layer_graph[parent1].output_buffer;
          int size0 = layer_graph[parent0].o * layer_graph[parent0].ho * layer_graph[parent0].wo;
          int size1 = layer_graph[parent1].o * layer_graph[parent1].ho * layer_graph[parent1].wo;
	  int new_buffer = next_buffer;
	  next_buffer++;

          layer_graph[l].layer_type = LAYER_DUMMY_CONCAT;

	  // buffer0 is LSB and buffer1 is MSB
	  int x = num_unified_buffers;
	  unified_buffers[x].valid = 1;
	  unified_buffers[x].num_buffers = 2;
	  unified_buffers[x].new_buffer = new_buffer;
	  unified_buffers[x].old_buffer[0] = old_buffer0;
	  unified_buffers[x].old_buffer[1] = old_buffer1;
	  unified_buffers[x].size[0] = size0;
          unified_buffers[x].size[1] = size1;
	  unified_buffers[x].from[0] = 0;
	  unified_buffers[x].to[0] = size0 - 1;
	  unified_buffers[x].from[1] = size0;
	  unified_buffers[x].to[1] = size0 + size1 - 1;
          num_unified_buffers++;
	}
      }
    }
  }

  // we now look into unifying two entries of unified buffers (three buffers into one): {b0, b1} and {b2, b1} where b2<b0 -> {b2, {b1|b0}}
  for (int x=0; x<num_unified_buffers; x++) {
    int buffer = unified_buffers[x].old_buffer[1];
    for (int y=x+1; y<num_unified_buffers; y++) {
      if ((unified_buffers[y].old_buffer[1] == buffer) && (unified_buffers[y].old_buffer[0] != unified_buffers[x].old_buffer[0])) {
	if (unified_buffers[y].size[0] < unified_buffers[x].size[0]) {
	  unified_buffers[x].num_buffers++;
	  unified_buffers[x].old_buffer[2] = unified_buffers[y].old_buffer[0];
	  unified_buffers[x].size[2] = unified_buffers[y].size[0];
	  unified_buffers[x].from[2] = unified_buffers[x].size[0] - unified_buffers[x].size[2];
	  unified_buffers[x].to[2] = unified_buffers[x].to[0];
	  unified_buffers[y].valid = 0;
	  break;
        }
      }
    }
  }
  num_buffers = next_buffer;

  // now we fix the input and output offsets in the graph
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_HLSINF) {
      int buffer = layer_graph[l].input_buffer[0];
      int offset;
      int total_size;
      int entry;
      if (shared_output_buffer(buffer, &offset, &entry, &total_size)) layer_graph[l].input_offset += offset;
      buffer = layer_graph[l].output_buffer;
      if (shared_output_buffer(buffer, &offset, &entry, &total_size)) {
	layer_graph[l].output_offset = offset;
      }
    }
  }

  // if the output buffer of a concat layer is made of two input buffers with output offsets then 
  // we need to add this offset to all layers taking this buffer as input
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_DUMMY_CONCAT) {
      int buffer = layer_graph[l].output_buffer;
      int parent0 = layer_graph[l].prev[0];
      int parent1 = layer_graph[l].prev[1];
      int offset = min(layer_graph[parent0].output_offset, layer_graph[parent1].output_offset);
      for (int x=0; x<graph_size; x++) {
        if (layer_graph[x].layer_type == LAYER_HLSINF) {
          if (layer_graph[x].input_buffer[0] == buffer) layer_graph[x].input_offset += offset;
	}
      }
    }
  }

}

int find_not_visited_and_with_visited_parents() {
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type != LAYER_NONE) {
      if (layer_graph[l].visited == 0) {
	 int all_parents_visited = 1;
	 for (int x=0; x<graph_size; x++) {
           if (layer_graph[x].layer_type != LAYER_NONE) {
	     for (int p=0; p<MAX_CHILDS; p++) {
	       if (layer_graph[x].next[p] == l) {
		 if (layer_graph[x].visited == 0) all_parents_visited = 0;
	       }
	     }
	   }
	 }
	 if (all_parents_visited) return l;
      }
    }
  }
  return -1;
}

int min(int a, int b) {return a<b?a:b;}

int num_parents(int l) {
  int num = 0;
  if (layer_graph[l].prev[0] != -1) num++;
  if (layer_graph[l].prev[1] != -1) num++;
  if (layer_graph[l].prev[2] != -1) num++;
  return num;
}

void draw_graph() {

  // first thing is to assign a column to each layer, a child of a single-child parent has the same column as its parent
  // if a layer has two childs, the first has the same column as its parent and the second child has a new one
  for (int l=0; l<graph_size; l++) {layer_graph[l].visited = 0; layer_graph[l].column = -1;}
  int next_column = 0;
  int processed_layers = 0;
  for (int l=0; l<graph_size; l++) if (layer_graph[l].layer_type == LAYER_NONE) processed_layers++;
  while (processed_layers != graph_size) {
    int l = find_not_visited_and_with_visited_parents();
    if (l==-1) {printf("unable to find layer to draw graph\n"); exit(1);}

    if (layer_graph[l].column == -1) {layer_graph[l].column = next_column; next_column++;}
    else if (num_parents(l) == 2) layer_graph[l].column = min(layer_graph[layer_graph[l].prev[0]].column, layer_graph[layer_graph[l].prev[1]].column);

    if (layer_graph[l].column == -1) {layer_graph[l].column = next_column; next_column++;}
    if (layer_graph[l].next[0] != -1) layer_graph[layer_graph[l].next[0]].column = layer_graph[l].column;
    if (layer_graph[l].next[1] != -1) {layer_graph[layer_graph[l].next[1]].column = next_column; next_column++;}
    if (layer_graph[l].next[2] != -1) {layer_graph[layer_graph[l].next[2]].column = next_column; next_column++;}
    layer_graph[l].visited = 1;
    processed_layers++;
  }

  // now we draw the graph
  for (int l=0; l<graph_size; l++) layer_graph[l].visited = 0;
  processed_layers = 0;
  for (int l=0; l<graph_size; l++) if (layer_graph[l].layer_type == LAYER_NONE) processed_layers++;
  while (processed_layers != graph_size) {
    int l = find_not_visited_and_with_visited_parents();
    if (l==-1) {printf("unable to find layer to draw graph\n"); exit(1);}
   // for (int x=0; x < layer_graph[l].column; x++) printf("               ");
    printf("%-10s (%d)\n", get_layer_name(layer_graph[l].layer_type), l);
    layer_graph[l].visited = 1;
    processed_layers++;
  }
}

void create_new_layers() {

  // variables to link the layers
  Layer *prev_layer = nullptr;
//  layer *parent_layer;

  // the graph is swept in the layer dependences order, so we use the visited field to guarantee this order
  for (int l=0; l<graph_size; l++) layer_graph[l].visited = 0;
  int processed_layers = 0;
  for (int l=0; l<graph_size; l++) if (layer_graph[l].layer_type == LAYER_NONE) processed_layers++;

  while (processed_layers != graph_size) {

    // let's get the next layer to build
    int l = find_not_visited_and_with_visited_parents();
    if (l==-1) {printf("unable to find layer in dependence's order\n"); exit(1);}

    // parent layers for this layer  
    int id_parent0 = layer_graph[l].prev[0];
    int id_parent1 = layer_graph[l].prev[1];
    int id_parent2 = layer_graph[l].prev[2];
    Layer *layer_parent0 = layer_graph[id_parent0].final_layer;
    Layer *layer_parent1 = (id_parent1 != -1) ? layer_graph[id_parent1].final_layer : nullptr;
    Layer *layer_parent2 = (id_parent2 != -1) ? layer_graph[id_parent2].final_layer : nullptr;

    // we avoid seg fault if the prev layer is a dummy concat (which does not generate any parent layer), we simply use the previous one
    if (layer_parent0 == nullptr) layer_parent0 = prev_layer; else prev_layer = layer_parent0;
   
    if (layer_graph[l].layer_type == LAYER_RELU) {
      layer_graph[l].final_layer = ReLu(layer_parent0);
    } else if (layer_graph[l].layer_type == LAYER_LEAKY_RELU) {
      LActivation *layer = (LActivation *)layer_graph[l].layer;
      layer_graph[l].final_layer = LeakyReLu(layer_parent0, layer->params[0]);
    } else if (layer_graph[l].layer_type == LAYER_SOFTMAX) {
      layer_graph[l].final_layer = Softmax(layer_parent0);
    } else if (layer_graph[l].layer_type == LAYER_SIGMOID) {
      layer_graph[l].final_layer = Sigmoid(layer_parent0);
    } else if (layer_graph[l].layer_type == LAYER_SOFTPLUS) {
      layer_graph[l].final_layer = Softplus(layer_parent0);
    } else if (layer_graph[l].layer_type == LAYER_TANH) {
      layer_graph[l].final_layer = Tanh(layer_parent0);
    } else if (layer_graph[l].layer_type == LAYER_LINEAR) {
      LActivation *layer = (LActivation *)layer_graph[l].layer;
      layer_graph[l].final_layer = Linear(layer_parent0, layer->params[0]); 
    } else if (layer_graph[l].layer_type == LAYER_MAX_POOL) {
      LMaxPool *layer = (LMaxPool *)layer_graph[l].layer;
      if (layer->pd->padding =="custom") {
        layer_graph[l].final_layer = new LMaxPool(layer_parent0, layer->pd->ksize, layer->pd->stride, layer->pd->pad, "", DEV_CPU, 0);
      } else {
	layer_graph[l].final_layer = new LMaxPool(layer_parent0, layer->pd->ksize, layer->pd->stride, layer->pd->padding, "", DEV_CPU, 0);
     }
    } else if (layer_graph[l].layer_type == LAYER_AVG_POOL) {
      LAveragePool *layer = (LAveragePool *)layer_graph[l].layer;
      layer_graph[l].final_layer = new LAveragePool(layer_parent0, layer->pd->ksize, layer->pd->stride, layer->pd->pad, "", DEV_CPU, 0); 
    } else if (layer_graph[l].layer_type == LAYER_RESHAPE) {
      LReshape *layer = (LReshape *)layer_graph[l].layer;
      long int elements = 1;
      for (int i = 1; i < layer->ls.size(); i++) elements = elements * layer->ls[i];
      if (layer->ls[1] == elements && layer->ls.size() < 3 ) {
        layer_graph[l].final_layer = Reshape(layer_parent0, { -1 });
      } else {
        vector<int> shape;
        for (int i = 1; i < layer->ls.size(); i++) shape.push_back(layer->ls[i]);
        layer_graph[l].final_layer = Reshape(layer_parent0, shape);
      }
    } else if (layer_graph[l].layer_type == LAYER_RESIZE) {
      LResize *layer = (LResize *)layer_graph[l].layer;
      layer_graph[l].final_layer = new LResize(layer_parent0, layer->new_shape, layer->reshape, layer->da_mode, layer->cval, layer->coordinate_transformation_mode, 
                                                              layer->name, layer->dev, 0);
    } else if (layer_graph[l].layer_type == LAYER_DENSE) {
      LDense *layer = (LDense *)layer_graph[l].layer;
      layer_graph[l].final_layer = Dense(layer_parent0, layer->ndim);
    } else if (layer_graph[l].layer_type == LAYER_CONCAT) {
      LConcat *layer = (LConcat *)layer_graph[l].layer;
      vector<Layer *> parent;
      parent.push_back(layer_parent0);
      parent.push_back(layer_parent1);
      if (layer_parent2 != nullptr) parent.push_back(layer_parent2);
      layer_graph[l].final_layer = Concat(parent, layer->axis, "");
    } else if (layer_graph[l].layer_type == LAYER_EXPAND) {
      LExpand *layer = (LExpand *)layer_graph[l].layer;
      layer_graph[l].final_layer = Expand(layer_parent0, layer->size, "");
    } else if (layer_graph[l].layer_type == LAYER_SELECT) {
      LSelect *layer = (LSelect *)layer_graph[l].layer;
      layer_graph[l].final_layer = Slice(layer_parent0, layer->sd->indices, "");
    } else if (layer_graph[l].layer_type == LAYER_MULT) {
      LMult *layer = (LMult *)layer_graph[l].layer;
      // two cases, one parent -> multiply by a constant, else (2 parents) -> multiply two tensors
      if (id_parent1 == -1) {
        // only one parent
        layer_graph[l].final_layer = Mult(layer_parent0, layer->val);
      } else {
        // two parents
        vector<Layer *> parent;
        parent.push_back(layer_parent0);
        parent.push_back(layer_parent1);
        vector<Layer *> operators = expand_broadcast(parent);
        layer_graph[l].final_layer = Mult(operators[0], operators[1]);
      }
    } else if (layer_graph[l].layer_type == LAYER_DIV) {
      LDiv *layer = (LDiv *)layer_graph[l].layer;
      layer_graph[l].final_layer = Div(layer_parent0, layer->val);
    } else if (layer_graph[l].layer_type == LAYER_DIFF) {
      LDiff *layer = (LDiff *)layer_graph[l].layer;
      if (layer->left) {
        layer_graph[l].final_layer = new LDiff(layer_parent0, layer->val, "", DEV_CPU, layer->mem_level);
      } else {
        layer_graph[l].final_layer = new LDiff(layer->val, layer_parent0, "", DEV_CPU, layer->mem_level);
      }
    } else if (layer_graph[l].layer_type == LAYER_EXP) {
      layer_graph[l].final_layer = Exp(layer_parent0);
    } else if (layer_graph[l].layer_type == LAYER_PERMUTE) {
      vector<int> dims;
      dims.push_back(2);
      dims.push_back(1);
      dims.push_back(0);
      layer_graph[l].final_layer = Permute(layer_parent0, dims);
    } else if (layer_graph[l].layer_type == LAYER_ADD) {
      vector<Layer *> parent;
      parent.push_back(layer_parent0);
      parent.push_back(layer_parent1);
      if (layer_parent2 != nullptr) parent.push_back(layer_parent2);
      layer_graph[l].final_layer = Add(parent);
    } else if (layer_graph[l].layer_type == LAYER_CONST_OF_TENSOR) {
      LConstOfTensor *layer = (LConstOfTensor *)layer_graph[l].layer;
      layer_graph[l].final_layer = ConstOfTensor(layer->const_tensor, layer->name);
    } else if (layer_graph[l].layer_type == LAYER_CLAMP) {
      LClamp *layer = (LClamp *)layer_graph[l].layer;
      layer_graph[l].final_layer = Clamp(layer_parent0, layer->min, layer->max);
    } else if (layer_graph[l].layer_type == LAYER_PAD) {
      LPad *layer = (LPad *)layer_graph[l].layer;
      layer_graph[l].final_layer = Pad(layer_parent0, layer->padding, layer->constant, "");
    } else if (layer_graph[l].layer_type == LAYER_UPSAMPLING) {
      LUpSampling *layer = (LUpSampling *)layer_graph[l].layer;
      layer_graph[l].final_layer = UpSampling(layer_parent0, layer->size, layer->interpolation);
    } else if (layer_graph[l].layer_type == LAYER_BATCH_NORM) {
      LBatchNorm *layer = (LBatchNorm *)layer_graph[l].layer;
      layer_graph[l].final_layer = BatchNormalization(layer_parent0, layer->momentum, layer->epsilon, layer->affine, "");
    } else if (layer_graph[l].layer_type == LAYER_DROP_OUT) {
      LDropout *layer = (LDropout *)layer_graph[l].layer;
      layer_graph[l].final_layer = Dropout(layer_parent0, layer->df, layer->iw, layer->name);
    } else if (layer_graph[l].layer_type == LAYER_CONV) {
      LConv *layer = (LConv *)layer_graph[l].layer;
      layer_graph[l].final_layer = new LConv(layer_parent0, layer->cd->filters, layer->cd->kernel_size, layer->cd->strides, layer->cd->padding, layer->cd->pads, layer->cd->groups, layer->cd->dilation_rate, layer->cd->use_bias, "", DEV_CPU, layer->cd->mem_level);
    } else if (layer_graph[l].layer_type == LAYER_HLSINF) {
      // two cases, one with add (two parents) and one with no add (one parent)
      if (id_parent1 != -1) {
        // two parents
        vector<Layer *> parent;
        parent.push_back(layer_parent0);
        parent.push_back(layer_parent1); 
        layer_graph[l].final_layer = new LHLSinf(parent, layer_graph[l].hi, layer_graph[l].wi, layer_graph[l].i, layer_graph[l].o,
                                                                                      layer_graph[l].kh, layer_graph[l].kw, layer_graph[l].sh, layer_graph[l].sw,
										      layer_graph[l].pt, layer_graph[l].pb, 
                                                                                      layer_graph[l].pl, layer_graph[l].pr, layer_graph[l].apply_relu, layer_graph[l].relu_factor,
                                                                                      layer_graph[l].apply_cliping, layer_graph[l].min_clip, layer_graph[l].max_clip,
                                                                                      layer_graph[l].apply_shift, layer_graph[l].pos_shift, layer_graph[l].dir_shift,
                                                                                      layer_graph[l].apply_stm, layer_graph[l].apply_maxp, layer_graph[l].apply_avgp,
                                                                                      layer_graph[l].apply_bn, layer_graph[l].apply_bn_relu, layer_graph[l].bn_relu_factor,
                                                                                      layer_graph[l].apply_add, layer_graph[l].apply_add_relu, layer_graph[l].upscale_factor,
                                                                                      layer_graph[l].apply_dense, layer_graph[l].apply_weight_buffer, layer_graph[l].first_row_weight_buffer,
                                                                                      layer_graph[l].input_offset, layer_graph[l].output_offset, 
										      layer_graph[l].sz_layer_name, DEV_CPU, 0);
      } else {
        // one parent
        layer_graph[l].final_layer = new LHLSinf(layer_parent0, layer_graph[l].hi, layer_graph[l].wi, layer_graph[l].i, layer_graph[l].o,
                                                                                      layer_graph[l].kh, layer_graph[l].kw, layer_graph[l].sh, layer_graph[l].sw,
										      layer_graph[l].pt, layer_graph[l].pb, 
                                                                                      layer_graph[l].pl, layer_graph[l].pr, layer_graph[l].apply_relu, layer_graph[l].relu_factor,
                                                                                      layer_graph[l].apply_cliping, layer_graph[l].min_clip, layer_graph[l].max_clip,
                                                                                      layer_graph[l].apply_shift, layer_graph[l].pos_shift, layer_graph[l].dir_shift,
                                                                                      layer_graph[l].apply_stm, layer_graph[l].apply_maxp, layer_graph[l].apply_avgp,
                                                                                      layer_graph[l].apply_bn, layer_graph[l].apply_bn_relu, layer_graph[l].bn_relu_factor,
                                                                                      layer_graph[l].apply_add, layer_graph[l].apply_add_relu, layer_graph[l].upscale_factor,
                                                                                      layer_graph[l].apply_dense, layer_graph[l].apply_weight_buffer, layer_graph[l].first_row_weight_buffer,
                                                                                      layer_graph[l].input_offset, layer_graph[l].output_offset, 
										      layer_graph[l].sz_layer_name, DEV_CPU, 0);
      }
    } else if (layer_graph[l].layer_type == LAYER_INPUT) {
      layer_graph[l].final_layer = Input({layer_graph[l].i, layer_graph[l].hi, layer_graph[l].wi});
    } else if (layer_graph[l].layer_type == LAYER_TRANSFORM) {
      layer_graph[l].final_layer = Transform(layer_parent0, layer_graph[l].cpu2fpga, layer_graph[l].fpga2cpu, layer_graph[l].transform, 0);
    } else if (layer_graph[l].layer_type == LAYER_DUMMY_CONCAT) {
      LConcat *layer = (LConcat *)layer_graph[l].layer;
      vector<Layer *> parent;
      parent.push_back(layer_parent0);
      parent.push_back(layer_parent1);
      layer_graph[l].final_layer = new LDConcat(parent, layer->axis, "DConcat", DEV_CPU, 0);
    } else {
      printf("Error, layer type not recognized\n"); exit(1);
    }

    layer_graph[l].visited = 1;
    processed_layers++;
  }
}

void adapt_weights_new_model() {

  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_CONV) {
      printf("adapting layer %d\n", l);
      LConv *layer_dst = (LConv *) layer_graph[l].final_layer;
      LConv *layer_src = (LConv *) layer_graph[l].layer;

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
    }

    if (layer_graph[l].layer_type == LAYER_BATCH_NORM) {
      LBatchNorm *layer_dst = (LBatchNorm *) layer_graph[l].final_layer;
      LBatchNorm *layer_src = (LBatchNorm *) layer_graph[l].layer;

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
    }

    if (layer_graph[l].layer_type == LAYER_HLSINF) {
      LHLSinf *layer_dst = (LHLSinf *) layer_graph[l].final_layer;
      LConv *layer_src_conv = (LConv *) layer_graph[l].conv_layer;
      LDense *layer_src_dense = (LDense *) layer_graph[l].dense_layer;
      LBatchNorm *layer_src_bn = (LBatchNorm *) layer_graph[l].bn_layer;

      if (layer_graph[l].apply_conv == 0) {
	// we create the identity filter for conv
	layer_dst->filter = new Tensor({1, layer_graph[l].o, layer_graph[l].i, layer_graph[l].kh, layer_graph[l].kw});
	int GI = layer_graph[l].i / CPI;
	int GO = layer_graph[l].o / CPO;
        memset(layer_dst->filter->ptr, 0, sizeof(float) * layer_graph[l].kh * layer_graph[l].kw * layer_graph[l].i * layer_graph[l].o);
        for (int i=0; i < layer_graph[l].i; i++) {
          int gi = i / CPI;
          int cpi = i % CPI;
          int go = i / CPO;
          int cpo = i % CPO;
          int addr = (go * GI * CPO * CPI * layer_graph[l].kh * layer_graph[l].kw) + (gi * CPO * CPI * layer_graph[l].kh * layer_graph[l].kw) +
                     (cpo * CPI * layer_graph[l].kh * layer_graph[l].kw) + (cpi * layer_graph[l].kh * layer_graph[l].kw) + (0 * layer_graph[l].kw) + 0;
          layer_dst->filter->ptr[addr] = 1;
        }

	// now the bias
	layer_dst->bias = new Tensor({layer_graph[l].o});
	for (int o=0; o<layer_graph[l].o; o++) layer_dst->bias->ptr[o] = 0;
      } else {

        if (layer_src_conv != NULL) {
          collectTensor(layer_src_conv, "param", 0);
          filter_IHW_to_GIHWCPI(layer_src_conv->cd->K, layer_dst->filter);
          distributeTensor(layer_dst, "param", 0);
          collectTensor(layer_src_conv, "param", 1);
          if (layer_src_conv->cd->use_bias) tensor_padded(layer_src_conv->cd->bias, layer_dst->bias); else memset(layer_dst->bias->ptr, 0, sizeof(float) * layer_dst->bias->size);
          distributeTensor(layer_dst, "param", 1);
        }

        if (layer_src_bn != NULL) {
          collectTensor(layer_src_bn, "param", 0);
          collectTensor(layer_src_bn, "param", 1);
          collectTensor(layer_src_bn, "param", 2);
          collectTensor(layer_src_bn, "param", 3);
          get_batch_norm_values(layer_src_conv->cd->O->shape[1], layer_src_bn->mean, layer_src_bn->variance, layer_src_bn->bn_g, layer_src_bn->bn_b, layer_dst->batch_norm_values);
         distributeTensor(layer_dst, "param", 2);
        }
      }

      if (layer_src_dense != NULL) {
        collectTensor(layer_src_dense, "param", 0);
        dense_to_conv(layer_src_dense->W->ptr, layer_src_dense->W->shape[0], layer_src_dense->W->shape[1], layer_dst->filter->ptr, layer_dst->Ichannels, layer_dst->Ochannels, layer_dst->KH, layer_dst->KW);
        if (hlsinf_filter_format == HLSINF_FP32) {
          layer_dst->filter->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer_dst->filter->size*sizeof(float));
          fpga_copy_memory_to_fpga(layer_dst->filter->ptr, layer_dst->filter->fpga_ptr, layer_dst->filter->size*sizeof(float));
        } else if (hlsinf_filter_format == HLSINF_API8) {
          layer_dst->filter->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer_dst->filter->size * sizeof(ap_int<8>));
          fpga_copy_memory_to_fpga_and_format(layer_dst->filter->ptr, (cl::Buffer *)layer_dst->filter->fpga_ptr, layer_dst->filter->size, HLSINF_FP32, HLSINF_API8);
        } else if (hlsinf_filter_format == HLSINF_APF_8_4) {
          layer_dst->filter->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer_dst->filter->size * sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>));
          fpga_copy_memory_to_fpga_and_format(layer_dst->filter->ptr, (cl::Buffer *)layer_dst->filter->fpga_ptr, layer_dst->filter->size, HLSINF_FP32, HLSINF_APF_8_4);
        } else if (hlsinf_filter_format == HLSINF_APF_16_8) {
          layer_dst->filter->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer_dst->filter->size * sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>));
          fpga_copy_memory_to_fpga_and_format(layer_dst->filter->ptr, (cl::Buffer *)layer_dst->filter->fpga_ptr, layer_dst->filter->size, HLSINF_FP32, HLSINF_APF_16_8);
        } else {
          printf("Error (HLSinf forward), filter format not supported\n");
          exit(1);
        }

        distributeTensor(layer_dst, "param", 0);
        if (layer_src_dense->use_bias) {
          collectTensor(layer_src_dense, "param", 1);
          tensor_padded(layer_src_dense->bias, layer_dst->bias);
          if (hlsinf_bias_format == HLSINF_FP32) {
            layer_dst->bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer_dst->bias->size*sizeof(float));
            fpga_copy_memory_to_fpga(layer_dst->bias->ptr, layer_dst->bias->fpga_ptr, layer_dst->bias->size*sizeof(float));
          } else if (hlsinf_bias_format == HLSINF_API32) {
            layer_dst->bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer_dst->bias->size*sizeof(ap_int<32>));
            fpga_copy_memory_to_fpga_and_format(layer_dst->bias->ptr, (cl::Buffer *)layer_dst->bias->fpga_ptr, layer_dst->bias->size, HLSINF_FP32, HLSINF_API32);
          } else if (hlsinf_bias_format == HLSINF_APF_8_4) {
            layer_dst->bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer_dst->bias->size*sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>));
            fpga_copy_memory_to_fpga_and_format(layer_dst->bias->ptr, (cl::Buffer *)layer_dst->bias->fpga_ptr, layer_dst->bias->size, HLSINF_FP32, HLSINF_APF_8_4);
          } else if (hlsinf_bias_format == HLSINF_APF_16_8) {
            layer_dst->bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer_dst->bias->size*sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>));
            fpga_copy_memory_to_fpga_and_format(layer_dst->bias->ptr, (cl::Buffer *)layer_dst->bias->fpga_ptr, layer_dst->bias->size, HLSINF_FP32, HLSINF_APF_16_8);
          } else {
            printf("Error (HLSinf forward), bias format not supported\n");
            exit(1);
          }
          distributeTensor(layer_dst, "param", 1);
        }
      }
    }

    if (layer_graph[l].layer_type == LAYER_DENSE) {
      LDense *layer_dst = (LDense *) layer_graph[l].final_layer;
      LDense *layer_src = (LDense *) layer_graph[l].layer;

      collectTensor(layer_src, "param", 0);
      tensor_padded(layer_src->W, layer_dst->W);
      distributeTensor(layer_dst, "param", 0);
      collectTensor(layer_src, "param", 1);
      tensor_padded(layer_src->bias, layer_dst->bias);
      distributeTensor(layer_dst, "param", 1);
    } 
    
    if (layer_graph[l].layer_type == LAYER_DROP_OUT) {
      LDropout *layer_dst = (LDropout *) layer_graph[l].final_layer;
      LDropout *layer_src = (LDropout *) layer_graph[l].layer;
      tensor_padded(layer_src->mask, layer_dst->mask);
    }    
  }
}

void allocate_fpga_buffers() {

  // weights, bias, bn buffers for HLSinf layers and output buffers for transform layers
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_HLSINF) {
      // weights
      LHLSinf *layer = (LHLSinf *)layer_graph[l].final_layer;
      if (hlsinf_filter_format == HLSINF_FP32) {
        // We simply create the buffer and copy the tensor into the buffer (no data type conversion needed)
        layer->filter->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer->filter->size*sizeof(float));
        fpga_copy_memory_to_fpga(layer->filter->ptr, layer->filter->fpga_ptr, layer->filter->size*sizeof(float));
      } else if (hlsinf_filter_format == HLSINF_API8) {
        // Data conversion needed (FP32->API8)
        layer->filter->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer->filter->size * sizeof(ap_int<8>));
        fpga_copy_memory_to_fpga_and_format(layer->filter->ptr, (cl::Buffer *)layer->filter->fpga_ptr, layer->filter->size, HLSINF_FP32, HLSINF_API8);
      } else if (hlsinf_filter_format == HLSINF_APF_8_4) {
        // Data conversion needed (FP32->APF<8,4,AP_RND_ZERO,AP_SAT>)
        layer->filter->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer->filter->size * sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>));
        fpga_copy_memory_to_fpga_and_format(layer->filter->ptr, (cl::Buffer *)layer->filter->fpga_ptr, layer->filter->size, HLSINF_FP32, HLSINF_APF_8_4);
      } else if (hlsinf_filter_format == HLSINF_APF_16_8) {
        // Data conversion needed (FP32->APF<16,8,AP_RND_ZERO,AP_SAT>)
        layer->filter->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer->filter->size * sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>));
        fpga_copy_memory_to_fpga_and_format(layer->filter->ptr, (cl::Buffer *)layer->filter->fpga_ptr, layer->filter->size, HLSINF_FP32, HLSINF_APF_16_8);
      } else if (hlsinf_filter_format == HLSINF_APF_32_16) {
        // Data conversion needed (FP32->APF<16,8,AP_RND_ZERO,AP_SAT>)
        layer->filter->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer->filter->size * sizeof(ap_fixed<32,16>));
        fpga_copy_memory_to_fpga_and_format(layer->filter->ptr, (cl::Buffer *)layer->filter->fpga_ptr, layer->filter->size, HLSINF_FP32, HLSINF_APF_32_16);
      } else {
        printf("Error, filter format not supported\n");
        exit(1);
      }

      // bias
      if (hlsinf_bias_format == HLSINF_FP32) {
        // No need for data conversion (FP32->FP32), we allocate the buffer and copy the bias tensor there
        layer->bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer->bias->size*sizeof(float));
        fpga_copy_memory_to_fpga(layer->bias->ptr, layer->bias->fpga_ptr,layer->bias->size*sizeof(float));
      } else if (hlsinf_bias_format == HLSINF_API32) {
        // Data conversion needed to API32 (FP32->API32)
        layer->bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer->bias->size*4);
        fpga_copy_memory_to_fpga_and_format(layer->bias->ptr, (cl::Buffer *)layer->bias->fpga_ptr,layer->bias->size, HLSINF_FP32, HLSINF_API32);
      } else if (hlsinf_bias_format == HLSINF_APF_8_4) {
        // Data conversion needed to APF_8_4 (FP32->APF<8,4,AP_RND_ZERO,AP_SAT>)
        layer->bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer->bias->size*sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>));
        fpga_copy_memory_to_fpga_and_format(layer->bias->ptr, (cl::Buffer *)layer->bias->fpga_ptr, layer->bias->size, HLSINF_FP32, HLSINF_APF_8_4);
      } else if (hlsinf_bias_format == HLSINF_APF_16_8) {
        // Data conversion needed to APF_8_4 (FP32->APF<16,8,AP_RND_ZERO,AP_SAT>)
        layer->bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer->bias->size*sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>));
        fpga_copy_memory_to_fpga_and_format(layer->bias->ptr, (cl::Buffer *)layer->bias->fpga_ptr, layer->bias->size, HLSINF_FP32, HLSINF_APF_16_8);
      } else if (hlsinf_bias_format == HLSINF_APF_32_16) {
        // Data conversion needed to APF_8_4 (FP32->APF<16,8,AP_RND_ZERO,AP_SAT>)
        layer->bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer->bias->size*sizeof(ap_fixed<32,16>));
        fpga_copy_memory_to_fpga_and_format(layer->bias->ptr, (cl::Buffer *)layer->bias->fpga_ptr, layer->bias->size, HLSINF_FP32, HLSINF_APF_32_16);
      } else {
        printf("Erro, bias format not supported\n");
        exit(1);
      }

      // bn
      if (hlsinf_bn_support) {
        // BatchNorm values assumed to be always in FP32 (might not be the case!)
        layer->batch_norm_values->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, layer->batch_norm_values->size*sizeof(float));
        fpga_copy_memory_to_fpga(layer->batch_norm_values->ptr, layer->batch_norm_values->fpga_ptr, layer->batch_norm_values->size*sizeof(float));
      }

    } else if (layer_graph[l].layer_type == LAYER_TRANSFORM) {
      LTransform *layer = (LTransform *)layer_graph[l].final_layer;
      if (layer->copy_cpu_to_fpga) {
        int size_in_bytes;
        if (hlsinf_output_format == HLSINF_FP32) size_in_bytes = 4;
        else if (hlsinf_input_format == HLSINF_API8) size_in_bytes = 1;
        else if (hlsinf_input_format == HLSINF_APF_8_4) size_in_bytes = sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>);
        else if (hlsinf_input_format == HLSINF_APF_16_8) size_in_bytes = sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>);
        else if (hlsinf_input_format == HLSINF_APF_32_16) size_in_bytes = sizeof(ap_fixed<32,16>);
        else if (hlsinf_input_format == HLSINF_APUI8) size_in_bytes = 1;
        else {printf("Error, input format not supported\n"); exit(1);}
        layer->output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, layer->output->size * size_in_bytes);
      }
    }
  }

  // output buffers for HLSinf layers
  // some output buffers might be shared (due to dummy concat layers)

  int size_in_bytes;
  if (hlsinf_output_format == HLSINF_FP32) size_in_bytes = 4;
  else if (hlsinf_output_format == HLSINF_API8) size_in_bytes = 1;
  else if (hlsinf_output_format == HLSINF_APF_8_4) size_in_bytes = sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>);
  else if (hlsinf_output_format == HLSINF_APF_16_8) size_in_bytes = sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>);
  else if (hlsinf_output_format == HLSINF_APF_32_16) size_in_bytes = sizeof(ap_fixed<32,16>);
  else if (hlsinf_output_format == HLSINF_APUI8) size_in_bytes = 1;
  else {printf("Error, output format not supported\n"); exit(1);}

  // we first allocate shared buffers
  for (int x=0; x < num_unified_buffers; x++) {
    if (unified_buffers[x].valid) {
      int size = 0;
      for (int p=0; p<unified_buffers[x].num_buffers; p++) size += unified_buffers[x].size[p];
      unified_buffers[x].ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, size * size_in_bytes);
    }
  }

  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_HLSINF) {
      LHLSinf *layer = (LHLSinf *)layer_graph[l].final_layer;
      int offset;
      int total_size;
      int entry;
      if (shared_output_buffer(layer_graph[l].output_buffer, &offset, &entry, &total_size)) {
	layer->output->fpga_ptr = unified_buffers[entry].ptr;
      } else {
	int size = ((LHLSinf *)layer_graph[l].final_layer)->output->size;
        layer->output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, size * size_in_bytes);
      }
    }
  }

  // finally we set fpga_ptr pointers in dummy concat layers
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_DUMMY_CONCAT) {
      LHLSinf *layer = (LHLSinf *)layer_graph[l].final_layer;
      layer->output->fpga_ptr = layer->input->fpga_ptr;
    }
  }

/*
  // output buffers for HLSinf layers
  // some output buffers might be shared (due to concat layers being optimized)
  for (int l=0; l<graph_size; l++) layer_graph[l].output_fpga_buffer_created = 0;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_HLSINF) {
      if (!layer_graph[l].output_fpga_buffer_created) {
        LHLSinf *layer = (LHLSinf *)layer_graph[l].final_layer;
        int size = layer->output->size;
	for (int x=l+1; x<graph_size; x++) {
          if (layer_graph[x].layer_type == LAYER_HLSINF) {
	    if (layer_graph[l].output_buffer == layer_graph[x].output_buffer) {
	      size += ((LHLSinf *)layer_graph[x].final_layer)->output->size;
	    }
	  }
	}
        int size_in_bytes;
        if (hlsinf_output_format == HLSINF_FP32) size_in_bytes = 4;
        else if (hlsinf_output_format == HLSINF_API8) size_in_bytes = 1;
        else if (hlsinf_output_format == HLSINF_APF_8_4) size_in_bytes = sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>);
        else if (hlsinf_output_format == HLSINF_APF_16_8) size_in_bytes = sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>);
        else if (hlsinf_output_format == HLSINF_APF_32_16) size_in_bytes = sizeof(ap_fixed<32,16>);
        else if (hlsinf_output_format == HLSINF_APUI8) size_in_bytes = 1;
        else {printf("Error, output format not supported\n"); exit(1);}
        layer->output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, size * size_in_bytes);
        layer_graph[l].output_fpga_buffer_created = 1;
        for (int x=l+1; x<graph_size; x++) {
          if (layer_graph[x].layer_type == LAYER_HLSINF) {
            if (layer_graph[l].output_buffer == layer_graph[x].output_buffer) {
              ((LHLSinf *)layer_graph[x].final_layer)->output->fpga_ptr = layer->output->fpga_ptr;
	      ((LHLSinf *)layer_graph[x].final_layer)->output->size = size;
	      layer_graph[x].output_fpga_buffer_created = 1;
            }
          }
        }
      }
    }
  }*/
}

void allocate_weight_buffers() {

  int remaining_rows_weight_buffer = hlsinf_weight_buffer;
  int first_row_weight_buffer_to_assign = 0;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type == LAYER_HLSINF) {
      int use_weight_buffer;
      int first_row_weight_buffer;
      int rows_weight_buffer_needed = (layer_graph[l].i / hlsinf_cpi) * (layer_graph[l].o / hlsinf_cpo);
      if ((remaining_rows_weight_buffer - rows_weight_buffer_needed) > 0) {
        use_weight_buffer = 1;
        first_row_weight_buffer = first_row_weight_buffer_to_assign;
        first_row_weight_buffer_to_assign = first_row_weight_buffer_to_assign + rows_weight_buffer_needed;
        remaining_rows_weight_buffer = remaining_rows_weight_buffer - rows_weight_buffer_needed;
      } else {
        use_weight_buffer = 0;
        first_row_weight_buffer = 0;
      }
      LHLSinf *layer = (LHLSinf *)layer_graph[l].final_layer;
      layer->use_weight_buffer = use_weight_buffer;
      layer->first_row_weight_buffer = first_row_weight_buffer;
      printf("HLSinf layer, weight buffer: Layer needs %8d weight buffer rows -> use the buffer: %s, remaining rows: %7d\n", rows_weight_buffer_needed, use_weight_buffer?"yes":"no ", remaining_rows_weight_buffer);

    }
  }    
}

model toFPGA(model m_src, int kernel_version, int kernel_subversion) {

  printf("Model conversion to FPGA:\n");
  #ifdef FPGA_DEBUG
  printf("initializing fpga...\n");
  #endif
  fpga_init(kernel_version, kernel_subversion);
  CPI = hlsinf_cpi;
  CPO = hlsinf_cpo;
  printf("  - FPGA initialized\n");
	
  #ifdef FPGA_DEBUG
  printf("building network graph...\n");
  #endif
  int nl = build_network_graph(m_src);
  #ifdef FPGA_DEBUG
  print_network_graph();
  draw_graph();
  #endif
  printf("  - Network graph built, found %d layers\n", nl);

  #ifdef FPGA_DEBUG
  printf("removing leftover layers...\n");
  #endif
  int rl = remove_leftover_layers();
  #ifdef FPGA_DEBUG
  print_network_graph();
  draw_graph();
  #endif
  printf("  - removed leftover layers, %d layers removed\n", rl);

  #ifdef FPGA_DEBUG
  printf("applying HLSinf layers...\n");
  #endif
  apply_hlsinf_layers();
  #ifdef FPGA_DEBUG
  print_network_graph();
  draw_graph();
  #endif
  printf("  - aplied HLSinf layers\n");

  #ifdef FPGA_DEBUG
  printf("adding FPGA buffers...\n");
  #endif
  add_fpga_buffers();
  #ifdef FPGA_DEBUG
  print_network_graph();
  print_buffers_info();
  #endif
  printf("  - added FPGA buffers\n");

  #ifdef FPGA_DEBUG
  printf("removing concat layers...\n");
  #endif
  //remove_concat_layers();
  #ifdef FPGA_DEBUG
  print_network_graph();
  draw_graph();
  #endif
  printf("  - removed concat layers\n");

  #ifdef FPGA_DEBUG
  printf("unifying buffers...\n");
  #endif
  unify_buffers();
  #ifdef FPGA_DEBUG
  for (int x=0; x < num_unified_buffers; x++) {
    if (unified_buffers[x].valid) {
      printf("buffer: %d -> ", unified_buffers[x].new_buffer);
      for (int p=0; p<unified_buffers[x].num_buffers; p++) printf("%d (size %d; range %d - %d)) ", unified_buffers[x].old_buffer[p], unified_buffers[x].size[p], unified_buffers[x].to[p], unified_buffers[x].from[p]);
      printf("\n");
    }
  }
  #endif
  printf("  - buffers unified\n");

  #ifdef FPGA_DEBUG
  printf("adding Transform layers...\n");
  #endif
  add_transform_layers();
  #ifdef FPGA_DEBUG
  print_network_graph();
  draw_graph();
  #endif
  printf("  - added transform layers\n");

  #ifdef FPGA_DEBUG
  printf("creating layers for the new model...\n");
  #endif
  create_new_layers();
  printf("  - created new layers\n");

  // Now we create the new model
  #ifdef FPGA_DEBUG
  printf("building the new model...\n");
  #endif
  Net *net = new Net();

  // Let's get input and output layers
  vlayer input_layers;
  vlayer output_layers;
  for (int l=0; l<graph_size; l++) {
    if (layer_graph[l].layer_type != LAYER_NONE) {
      if (layer_graph[l].in_layer) input_layers.push_back(layer_graph[l].final_layer);
      if (layer_graph[l].out_layer) output_layers.push_back(layer_graph[l].final_layer);
    }
  }
  net = Model({ input_layers }, { output_layers });

  if (net->lout.size() == 1) build(net, nullptr, { "soft_cross_entropy" }, { "categorical_accuracy" }, CS_CPU(-1, "low_mem"), 0);
  else if (net->lout.size() == 2) build(net, nullptr, { "soft_cross_entropy", "soft_cross_entropy" }, { "categorical_accuracy", "categorical_accuracy"}, CS_CPU(-1, "low_mem"), 0);
  else {printf("Number of outputs (%d) not supported in toFPGA(), please extend support\n", net->lout.size()); exit(1);}
  printf("  - built model\n");

  #ifdef FPGA_DEBUG
  summary(net);
  #endif

  #ifdef FPGA_DEBUG
  printf("adapting weights and bias to the new model...\n");
  #endif
  adapt_weights_new_model();
  printf("  - adapted weights\n");

  // allocating FPGA buffers
  #ifdef FPGA_DEBUG
  printf("allocating FPGA buffers...\n");
  #endif
  allocate_fpga_buffers();
  printf("  - allocated FPGA buffers\n");

  // weight buffers
  #ifdef FPGA_DEBUG
  printf("allocating weight buffers...\n");
  #endif
  //allocate_weight_buffers();
  printf(" - allocated weight buffers\n");

  return net; 
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

