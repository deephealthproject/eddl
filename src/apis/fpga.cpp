/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "eddl/apis/eddl.h"
#include "eddl/utils.h"
#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/cpu/cpu_tensor.h"

#define DEBUG_VERBOSE

extern void fpga_reshape_kernel(ConvolDescriptor *src_D, ConvolDescriptor *D, int KW, int KH, int I, int O, int CPI, int CPO);
extern void _profile_fpga_tensor(Tensor *t);
using namespace std;

extern void build(eddl::model net, eddl::optimizer o, const vector<Loss *> &lo, const vector<Metric *> &me, CompServ *cs, bool init_weights);

////////////////////////////////////////////////////////
///// EDDL is a wrapper class to ease and define the API
////////////////////////////////////////////////////////

namespace eddl {


void get_fpga_model_params(Net * fpga_model) {
	int num_layers = fpga_model->layers.size();  // number of layers  
  FILE *fptr;
  fptr = fopen("/home/laumecha/params_skin_seg_inf.data","w");
  if(fptr == NULL)
  {
     printf("Error!");   
     exit(1);             
  }
  Layer *cl; // current layer pointer
  for(int l = 0; l < num_layers; l++) {
    cl = fpga_model->layers[l];
    if (LConv *dl = dynamic_cast<LConv *>(cl)) {
      LConv *layer_dst = (LConv *) cl;
      fprintf(fptr,"ENABLE %d CPU %d DET 1 %dx%dx%dx%d KH %d KW %d PT %d PB %d PL %d PR %d SH %d SW %d RELU %d RELU_FACTOR 0 STM %d MAXPOOL %d AVGPOOL %d ADD %d SHIFT %d DIRECTION_SHIFT %d POS_SHIFT %d CLIP %d MINCLIP %d MAXCLIP %d\n",
      1, 1, layer_dst->cd->I->shape[2] , layer_dst->cd->I->shape[3], layer_dst->cd->I->shape[1], layer_dst->cd->O->shape[1],
      layer_dst->cd->kr, layer_dst->cd->kc, layer_dst->cd->pads[0], layer_dst->cd->pads[1], layer_dst->cd->pads[2], layer_dst->cd->pads[3], layer_dst->cd->sr, layer_dst->cd->sc, /*&enable_relu*/ 0, /*&enable_stm*/ 0, /*&enable_maxpooling*/ 0,
       0, /*&enable_add*/ 0, /*&enable_shift*/ 0, /*&dir_shift*/ 0, /*&pos_shift*/ 0,0,0,0);

    } else if (LHLSinf *conv = dynamic_cast<LHLSinf *>(cl)) {
      LHLSinf *layer_dst = (LHLSinf *) cl;
      fprintf(fptr,"ENABLE %d CPU %d DET 1 %dx%dx%dx%d KH %d KW %d PT %d PB %d PL %d PR %d SH %d SW %d RELU %d RELU_FACTOR %f STM %d MAXPOOL %d AVGPOOL %d ADD %d SHIFT %d DIRECTION_SHIFT %d POS_SHIFT %d CLIP %d MINCLIP %d MAXCLIP %d\n",
      1, 1, layer_dst->H , layer_dst->W, layer_dst->Ichannels, layer_dst->Ochannels,
      layer_dst->KH, layer_dst->KW, layer_dst->PT, layer_dst->PB, layer_dst->PL, layer_dst->PR, layer_dst->SH, layer_dst->SW, 
      layer_dst->enable_relu , layer_dst->relu_factor, layer_dst->enable_stm, layer_dst->enable_maxp,
       layer_dst->enable_avgp, layer_dst->enable_add, layer_dst->enable_shift, /*&dir_shift*/ 0, layer_dst->pos_shift,0,0,0);

    }
  }
  fclose(fptr);
}

  void model_to_hls(model m_src) {

    int stream_id = 0;
    int filter_id = 0;
    int scale_id = 0;
    int B_id = 0;
    int mean_id = 0;
    int var_id = 0;

    // number of layers
    int num_layers = m_src->layers.size();

    int l=0;
    while (l < num_layers) {

      Layer *cl = m_src->layers[l];	    

      if (LInput *dl = dynamic_cast<LInput *>(cl)) {    // Input layer (read)
	// output stream
	int C = cl->input->shape[1];
	int H = cl->input->shape[2];
	int W = cl->input->shape[3];
	int DATA_WIDTH = 32;
	int out_stream = stream_id;
	printf("\n// read data\n");
	printf("hls::stream<ap_uint<%d * %d>> stream_%0d;\n", DATA_WIDTH, C, stream_id);
	printf("read<%d, %d, %d, %d>((ap_uint<%d * %d> *)ptr_in, stream_%0d);\n", C, H, W, DATA_WIDTH, C, DATA_WIDTH, out_stream);
	stream_id++;
      } else if (LDiv *dl = dynamic_cast<LDiv *>(cl)) { // Div layer
	int C = cl->input->shape[1];
	int H = cl->input->shape[2];
	int W = cl->input->shape[3];
	int DATA_WIDTH = 32;
	int in_stream = stream_id - 1;
	int out_stream = stream_id;
	int PE = 1;
	float div_factor = dl->val;
	printf("\n// div\n");
        printf("hls::stream<ap_uint<%d * %d>> stream_%0d;\n", DATA_WIDTH, C, stream_id);
        printf("div<%d, %d, %d, %d, %d>(stream_%0d, %f, stream_%0d);\n", C, H, W, DATA_WIDTH, PE, in_stream, div_factor, out_stream);
	stream_id++;
      } else if (LMult *dl = dynamic_cast<LMult *>(cl)) { // Div layer
        int C = cl->input->shape[1];
        int H = cl->input->shape[2];
        int W = cl->input->shape[3];
        int DATA_WIDTH = 32;
        int in_stream = stream_id - 1;
        int out_stream = stream_id;
        int PE = 1;
        float mul_factor = dl->val;
        printf("\n// mult\n");
        printf("hls::stream<ap_uint<%d * %d>> stream_%0d;\n", DATA_WIDTH, C, stream_id);
        printf("mul<%d, %d, %d, %d, %d>(stream_%0d, %f, stream_%0d);\n", C, H, W, DATA_WIDTH, PE, in_stream, mul_factor, out_stream);
        stream_id++;
      } else if (LDiff *dl = dynamic_cast<LDiff *>(cl)) { // Substract layer
        int C = cl->input->shape[1];
        int H = cl->input->shape[2];
        int W = cl->input->shape[3];
        int DATA_WIDTH = 32;
        int in_stream = stream_id - 1;
        int out_stream = stream_id;
        int PE = 1;
        float sub_factor = dl->val;
        printf("\n// sub\n");
        printf("hls::stream<ap_uint<%d * %d>> stream_%0d;\n", DATA_WIDTH, C, stream_id);
        printf("sub<%d, %d, %d, %d, %d>(stream_%0d, %f, stream_%0d);\n", C, H, W, DATA_WIDTH, PE, in_stream, sub_factor, out_stream);
        stream_id++;
      } else if (LMultiThreshold *dl = dynamic_cast<LMultiThreshold *>(cl)) { // MultiThreshold layer
        int C = cl->input->shape[1];
        int H = cl->input->shape[2];
        int W = cl->input->shape[3];
        int DATA_WIDTH = 32;
        int in_stream = stream_id - 1;
        int out_stream = stream_id;
        int PE = 1;
        printf("\n// multithreshold\n");
        printf("hls::stream<ap_uint<%d * %d>> stream_%0d;\n", DATA_WIDTH, C, stream_id);
        printf("multithreshold .... ?????;\n");
        stream_id++;
      } else if (LSum *dl = dynamic_cast<LSum *>(cl)) { // Add layer
        int C = cl->input->shape[1];
        int H = cl->input->shape[2];
        int W = cl->input->shape[3];
        int DATA_WIDTH = 32;
        int in_stream = stream_id - 1;
        int out_stream = stream_id;
        int PE = 1;
        float add_factor = dl->val;
        printf("\n// add\n");
        printf("hls::stream<ap_uint<%d * %d>> stream_%0d;\n", DATA_WIDTH, C, stream_id);
        printf("sub<%d, %d, %d, %d, %d>(stream_%0d, %f, stream_%0d);\n", C, H, W, DATA_WIDTH, PE, in_stream, add_factor, out_stream);
        stream_id++;
      } else if (LConv *dl = dynamic_cast<LConv *>(cl)) { // Convolution layer
        printf("\n// convolution layer\n");
        // parameters
        int C = cl->input->shape[1];
        int H = cl->input->shape[2];
        int W = cl->input->shape[3];
        int KH = dl->cd->kernel_size[0];
        int KW = dl->cd->kernel_size[1];
        int SH = dl->cd->strides[0];
        int SW = dl->cd->strides[1];
        int PT = dl->cd->pads[0];
        int PB = dl->cd->pads[1];
        int PL = dl->cd->pads[2];
        int PR = dl->cd->pads[3];
        int O = dl->output->shape[1];
        int DATA_WIDTH = 32;
        int in_stream = stream_id - 1;
        int out_stream = stream_id;
        int PE = 1;
        int II = 1;
        // filters
        printf("ap_uint<%0d> filter_%0d[%0d] = {", C * KH * KW, filter_id, O);
        collectTensor(dl, "param", 0);
        for (int o=0; o<O; o++) {
          unsigned value = 0;
          int bit_pos = 0;
          for (int c=0; c<C; c++) {
            for (int kh=0; kh<KH; kh++) {
              for (int kw=0; kw<KW; kw++) {
                int addr = (o * C * KH * KW) + (c * KH * KW) + (kh * KW) + kw;
                float filter_value = dl->cd->K->ptr[addr];
                if (filter_value == 1) value = value | (1 << bit_pos);
                bit_pos++;
                if (bit_pos == 32) {
                  bit_pos = 0;
                  printf("0x%x", value);
                  value = 0;
                }
              }
            }
          }
          if (bit_pos != 0) printf("0x%x", value);
          if (o!=O-1) printf(","); else printf("};");
        }
        printf("\n");
        printf("hls::stream<ap_uint<%d * %d>> stream_%0d;\n", DATA_WIDTH, C, stream_id);
        printf("conv_bipolar<%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d>(stream_%0d, filter_%0d, stream_%0d);\n", C, H, W, O, KH, KW, PT, PB, PL, PR, SH, SW, DATA_WIDTH, DATA_WIDTH, II, PE, in_stream, filter_id, out_stream);
        stream_id++;
	      filter_id++;
      } else if (LBatchNorm *dl = dynamic_cast<LBatchNorm *>(cl)) { // Batch Normalization
        printf("\n// Batch Normalization\n");
        // parameters
        int C = cl->input->shape[1];
        int H = cl->input->shape[2];
        int W = cl->input->shape[3];
        int DATA_WIDTH = 32;
        int PE = 1;
        int in_stream = stream_id - 1;
        int out_stream = stream_id;

        // scale, B, mean, and var
        printf("float scale_%0d[%0d] = {", scale_id, C);
        printf("};");
        printf("\n");
        printf("float B_%0d[%0d] = {", B_id, C);
        printf("};");
        printf("\n");
        printf("float mean_%0d[%0d] = {", mean_id, C);
        collectTensor(dl, "param", 0);
        for (int c=0; c<C; c++) {
          float value = dl->bn_mean->ptr[c];
          printf("%f", value);
          if (c!=C-1) printf(","); else printf("};");
        }
        printf("\n");
        printf("float var_%0d[%0d] = {", var_id, C);
        collectTensor(dl, "param", 0);
        for (int c=0; c<C; c++) {
          float value = dl->bn_var->ptr[c];
          printf("%f", value);
          if (c!=C-1) printf(","); else printf("};");
        }
        printf("\n");
        printf("hls::stream<ap_uint<%d * %d>> stream_%0d;\n", DATA_WIDTH, C, stream_id);
        printf("batch_normalization<%d, %d, %d, %d, %d>(stream_%0d, scale_%0d, B_%0d, mean_%0d, var_%0d, stream_%0d);\n", C, H, W, DATA_WIDTH, PE, in_stream, scale_id, B_id, mean_id, var_id, out_stream);
        stream_id++;
	      scale_id++;
        B_id++;
        mean_id++;
        var_id++;
      } else if (LMaxPool *dl = dynamic_cast<LMaxPool *>(cl)) { // Maxpool
        printf("\n// Maxpooling\n");
        // parameters
        int C = cl->input->shape[1];
        int H = cl->input->shape[2];
        int W = cl->input->shape[3];
        int DATA_WIDTH = 32;
        int PE = 1;
        int in_stream = stream_id - 1;
        int out_stream = stream_id;
        int KH = dl->pd->ksize[0];
        int KW = dl->pd->ksize[1];
        int SH = dl->pd->stride[0];
        int SW = dl->pd->stride[1];
        printf("hls::stream<ap_uint<%d * %d>> stream_%0d;\n", DATA_WIDTH, C, stream_id);
        printf("maxpool<%d, %d, %d, %d, %d, %d, %d, %d, %d>(stream_%0d, stream_%0d);\n", C, H, W, KH, KW, SH, SW, DATA_WIDTH, PE, in_stream, out_stream);
      } else {
	printf("Error, layer not supported\n"); exit(1);
      }

      l++;

    }
  }
    
  void show_weight_stats(model m) {

    #define MAX_STATS 10000
    int stats_entries = 0;
    struct {
      float filters[3][3];
      int num_entries;
    } stats[MAX_STATS];

    // number of layers
    int num_layers = m->layers.size();

    int l=0;
    while (l < num_layers) {

      Layer *cl = m->layers[l];	    

      if (LConv *dl = dynamic_cast<LConv *>(cl)) { // Convolution layer
        // parameters
        int C = cl->input->shape[1];
        int H = cl->input->shape[2];
        int W = cl->input->shape[3];
        int KH = dl->cd->kernel_size[0];
        int KW = dl->cd->kernel_size[1];
        int SH = dl->cd->strides[0];
        int SW = dl->cd->strides[1];
        int PT = dl->cd->pads[0];
        int PB = dl->cd->pads[1];
        int PL = dl->cd->pads[2];
        int PR = dl->cd->pads[3];
        int O = dl->output->shape[1];

        printf("\n// convolution layer (l=%d) %dx%dx%dx%d\n", l, C, KH, KW, O);

        collectTensor(dl, "param", 0);
        for (int c=0; c<C; c++) {
          printf("analyzing input I=%d: ", c);

          for (int o=0; o<O; o++) {

            // ponemos tres posiciones a -1
            //int addr = (o * C * KH * KW) + (c * KH * KW) + (0 * KW) + 0;
            //dl->cd->K->ptr[addr] = -1;
            //addr = (o * C * KH * KW) + (c * KH * KW) + (0 * KW) + 1;
            //dl->cd->K->ptr[addr] = -1;
            //addr = (o * C * KH * KW) + (c * KH * KW) + (0 * KW) + 2;
            //dl->cd->K->ptr[addr] = -1;
            //addr = (o * C * KH * KW) + (c * KH * KW) + (1 * KW) + 0;
            //dl->cd->K->ptr[addr] = -1;
            //addr = (o * C * KH * KW) + (c * KH * KW) + (1 * KW) + 1;
            //dl->cd->K->ptr[addr] = -1;
            //addr = (o * C * KH * KW) + (c * KH * KW) + (1 * KW) + 2;
            //dl->cd->K->ptr[addr] = -1;
            //addr = (o * C * KH * KW) + (c * KH * KW) + (2 * KW) + 0;
            //dl->cd->K->ptr[addr] = -1;
            //addr = (o * C * KH * KW) + (c * KH * KW) + (2 * KW) + 1;
            //dl->cd->K->ptr[addr] = -1;
            //addr = (o * C * KH * KW) + (c * KH * KW) + (2 * KW) + 2;
            //dl->cd->K->ptr[addr] = -1;

            int ss;
            int found = 0;
            for (int s=0; s<stats_entries; s++) {
              found = 1;
              ss = s;
              for (int kh=0; kh<KH; kh++) {
                for (int kw=0; kw<KW; kw++) {
                  int addr = (o * C * KH * KW) + (c * KH * KW) + (kh * KW) + kw;
                  float filter_value = dl->cd->K->ptr[addr];
                  if (stats[s].filters[kh][kw] != filter_value) found = 0;
                }
              }
              if (found) break;
            }
            if (found) {
              stats[ss].num_entries++;
            } else {
              ss = stats_entries;
              for (int kh=0; kh<KH; kh++) {
                for (int kw=0; kw<KW; kw++) {
                  int addr = (o * C * KH * KW) + (c * KH * KW) + (kh * KW) + kw;
                  float filter_value = dl->cd->K->ptr[addr];
                  stats[ss].filters[kh][kw] = filter_value;
                }
              }
              stats[ss].num_entries = 1;
              stats_entries++; 
            }
          }

          // we sort entries
          for (int s1 = 0; s1<stats_entries - 1; s1++) {
            for (int s2 = s1+1; s2<stats_entries; s2++) {
              if (stats[s1].num_entries < stats[s2].num_entries) {
                int aux = stats[s1].num_entries;
                stats[s1].num_entries = stats[s2].num_entries;
                stats[s2].num_entries = aux;
                for (int kh=0; kh<3; kh++) for (int kw=0; kw<3; kw++) {
                  float faux = stats[s1].filters[kh][kw];
                  stats[s1].filters[kh][kw] = stats[s2].filters[kh][kw];
                  stats[s2].filters[kh][kw] = faux;
                }
              }
            }
          }

          // total num entries
          int sum_num_entries = 0;
          for (int s=0; s<stats_entries; s++) sum_num_entries += stats[s].num_entries;
          printf("%d filters (sum entries %d) -> ", stats_entries, sum_num_entries);
          int x = stats_entries;
          if (x>3) x=3;
          for (int s=0; s<x; s++) {
            for (int kh=0; kh<3; kh++) for (int kw=0; kw<3; kw++) printf("%4.2f ", stats[s].filters[kh][kw]);
            printf(" -> %5d entries (%6.4f\%) ", stats[s].num_entries, 100.0 * (float)stats[s].num_entries / (float)sum_num_entries);
          }

          // top four filters percentage
          int top_four = 0;
          if (stats_entries > 0) top_four += stats[0].num_entries;
          if (stats_entries > 1) top_four += stats[1].num_entries;
          if (stats_entries > 2) top_four += stats[2].num_entries;
          if (stats_entries > 3) top_four += stats[3].num_entries;
          printf(" -> top four: %d entries (%6.4f)", top_four, 100.0 * (float)top_four / (float)sum_num_entries);
          printf("\n");

          stats_entries = 0;

        }

      }
    
      l++;

    }

  }

}//namespace
