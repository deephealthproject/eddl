/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>


#include <sys/time.h>

#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"
#include "eddl/hardware/cpu/cpu_tensor.h"

#define VERBOSE 0

float get_pixel(int b,int px,int py,int pz,ConvolDescriptor *D,int isize,int irsize) {

  if (VERBOSE)
    cout<<"pixel: "<<pz<<" "<<py<<" "<<px<<endl;
  //getchar();
  
  // Check boundaries of the window
  if (px<0) return 0.0;
  if (py<0) return 0.0;
  if (px>=D->ic) return 0.0;
  if (py>=D->ir) return 0.0;

  // Compute address from indices (row-major)
  unsigned int address = (b*isize) + (pz*irsize) + (py*D->ic) + px;
  return D->I->ptr[address];
}

void add_pixel(int b,int px,int py,int pz,ConvolDescriptor *D,int isize,int irsize,float val) {
  // Check boundaries of the window
  if (px<0) return;
  if (py<0) return;
  if (px>=D->ic) return;
  if (py>=D->ir) return;

  // Compute address from indices (row-major)
  unsigned int address = (b*isize) + (pz*irsize) + (py*D->ic) + px;
  D->ID->ptr[address]+=val;
}


void im2col(int b,ConvolDescriptor *D,float *ptrI,int col2im)
{
  _profile(_CPU_IM2COL, 0);
  int i,j,k;
  int pz,py,px,y,x;
  int ksize=D->kr*D->kc;
  int kr2=D->kr/2;
  int kc2=D->kc/2;

  if (kc2==0) kc2=-1;

  int orsize=D->r*D->c;

  int isize=D->ir*D->ic*D->iz;
  int irsize=D->ir*D->ic;

  k=0;
  py=-D->padrt;
  px=-D->padcl;

  for(j=0;j<D->matI.rows();j++) {
    k=j;

    if (VERBOSE){
    cout<<"======================"<<endl;
    cout<<j<<" "<<j/D->c<<" "<<j%D->c<<endl;
    cout<<"======================"<<endl;
    }
    if ((j!=0)&&((j%D->c)==0)) {
       if (VERBOSE) cout<<"change row"<<endl; 
      px=-D->padcl;
      py+=D->sr;
    }

    
    for(i=0;i<D->matI.cols();i++,k+=orsize) {
      pz=i/ksize;
      y=py+(i%ksize)/D->kc;
      x=px+(i%D->kc);

      if(col2im)
      add_pixel(b,x,y,pz,D,isize,irsize,ptrI[k]);
      else
      ptrI[k]=get_pixel(b,x,y,pz,D,isize,irsize);

    }
    px+=D->sc;
  if (VERBOSE)    getchar();
  }
    _profile(_CPU_IM2COL, 1);

    if (VERBOSE){
    getchar();
    for(int i=0;i<100;i++) cout<<ptrI[i]<<" ";

    getchar();
    }
}

int cpu_kernel_offset(int i, int o, int kh, int kw, int I, int O, int KH, int KW) {
  return (o * KW * KH * I) + (i * KW * KH) + (kh * KW) + kw;
}

int cpu_data_offset(int i, int h, int w, int H, int W) {
  return (i * W * H) + (h * W) + w;
}


void cpu_print_data(ConvolDescriptor *D, int KW, int KH, int I, int O, int W, int H) {

  float *ptr = D->K->ptr;
  // filtro 0 0
  printf("F[0][0]\n");
  float f00_00 = ptr[cpu_kernel_offset(0, 0, 0, 0, I, O, KH, KW)];
  float f00_01 = ptr[cpu_kernel_offset(0, 0, 0, 1, I, O, KH, KW)];
  float f00_02 = ptr[cpu_kernel_offset(0, 0, 0, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f00_00, f00_01, f00_02);
  float f00_10 = ptr[cpu_kernel_offset(0, 0, 1, 0, I, O, KH, KW)];
  float f00_11 = ptr[cpu_kernel_offset(0, 0, 1, 1, I, O, KH, KW)];
  float f00_12 = ptr[cpu_kernel_offset(0, 0, 1, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f00_10, f00_11, f00_12);
  float f00_20 = ptr[cpu_kernel_offset(0, 0, 2, 0, I, O, KH, KW)];
  float f00_21 = ptr[cpu_kernel_offset(0, 0, 2, 1, I, O, KH, KW)];
  float f00_22 = ptr[cpu_kernel_offset(0, 0, 2, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f00_20, f00_21, f00_22);
  // filtro 1 0
  printf("F[1][0]\n");
  float f10_00 = ptr[cpu_kernel_offset(1, 0, 0, 0, I, O, KH, KW)];
  float f10_01 = ptr[cpu_kernel_offset(1, 0, 0, 1, I, O, KH, KW)];
  float f10_02 = ptr[cpu_kernel_offset(1, 0, 0, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f10_00, f10_01, f10_02);
  float f10_10 = ptr[cpu_kernel_offset(1, 0, 1, 0, I, O, KH, KW)];
  float f10_11 = ptr[cpu_kernel_offset(1, 0, 1, 1, I, O, KH, KW)];
  float f10_12 = ptr[cpu_kernel_offset(1, 0, 1, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f10_10, f10_11, f10_12);
  float f10_20 = ptr[cpu_kernel_offset(1, 0, 2, 0, I, O, KH, KW)];
  float f10_21 = ptr[cpu_kernel_offset(1, 0, 2, 1, I, O, KH, KW)];
  float f10_22 = ptr[cpu_kernel_offset(1, 0, 2, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f10_20, f10_21, f10_22);
  // filtro 2 0
  printf("F[2][0]\n");
  float f20_00 = ptr[cpu_kernel_offset(2, 0, 0, 0, I, O, KH, KW)];
  float f20_01 = ptr[cpu_kernel_offset(2, 0, 0, 1, I, O, KH, KW)];
  float f20_02 = ptr[cpu_kernel_offset(2, 0, 0, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f20_00, f20_01, f20_02);
  float f20_10 = ptr[cpu_kernel_offset(2, 0, 1, 0, I, O, KH, KW)];
  float f20_11 = ptr[cpu_kernel_offset(2, 0, 1, 1, I, O, KH, KW)];
  float f20_12 = ptr[cpu_kernel_offset(2, 0, 1, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f20_10, f20_11, f20_12);
  float f20_20 = ptr[cpu_kernel_offset(2, 0, 2, 0, I, O, KH, KW)];
  float f20_21 = ptr[cpu_kernel_offset(2, 0, 2, 1, I, O, KH, KW)];
  float f20_22 = ptr[cpu_kernel_offset(2, 0, 2, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f20_20, f20_21, f20_22);

  ptr = D->I->ptr;
  // data 0
  printf("D[0]\n");
  float d0_00 = ptr[cpu_data_offset(0, 0, 0, H, W)];
  float d0_01 = ptr[cpu_data_offset(0, 0, 1, H, W)];
  float d0_02 = ptr[cpu_data_offset(0, 0, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d0_00, d0_01, d0_02);
  float d0_10 = ptr[cpu_data_offset(0, 1, 0, H, W)];
  float d0_11 = ptr[cpu_data_offset(0, 1, 1, H, W)];
  float d0_12 = ptr[cpu_data_offset(0, 1, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d0_10, d0_11, d0_12);
  float d0_20 = ptr[cpu_data_offset(0, 2, 0, H, W)];
  float d0_21 = ptr[cpu_data_offset(0, 2, 1, H, W)];
  float d0_22 = ptr[cpu_data_offset(0, 2, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d0_20, d0_21, d0_22);
  // data 1
  printf("D[1]\n");
  float d1_00 = ptr[cpu_data_offset(1, 0, 0, H, W)];
  float d1_01 = ptr[cpu_data_offset(1, 0, 1, H, W)];
  float d1_02 = ptr[cpu_data_offset(1, 0, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d1_00, d1_01, d1_02);
  float d1_10 = ptr[cpu_data_offset(1, 1, 0, H, W)];
  float d1_11 = ptr[cpu_data_offset(1, 1, 1, H, W)];
  float d1_12 = ptr[cpu_data_offset(1, 1, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d1_10, d1_11, d1_12);
  float d1_20 = ptr[cpu_data_offset(1, 2, 0, H, W)];
  float d1_21 = ptr[cpu_data_offset(1, 2, 1, H, W)];
  float d1_22 = ptr[cpu_data_offset(1, 2, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d1_20, d1_21, d1_22);
  // data 2
  float d2_00 = ptr[cpu_data_offset(2, 0, 0, H, W)];
  float d2_01 = ptr[cpu_data_offset(2, 0, 1, H, W)];
  float d2_02 = ptr[cpu_data_offset(2, 0, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d2_00, d2_01, d2_02);
  float d2_10 = ptr[cpu_data_offset(2, 1, 0, H, W)];
  float d2_11 = ptr[cpu_data_offset(2, 1, 1, H, W)];
  float d2_12 = ptr[cpu_data_offset(2, 1, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d2_10, d2_11, d2_12);
  float d2_20 = ptr[cpu_data_offset(2, 2, 0, H, W)];
  float d2_21 = ptr[cpu_data_offset(2, 2, 1, H, W)];
  float d2_22 = ptr[cpu_data_offset(2, 2, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d2_20, d2_21, d2_22);

  // bias 0
  ptr = D->bias->ptr;
  printf("BIAS[0]\n");
  float b0 = ptr[0];
  printf("%6.4f\n", b0);

  float pixel_out = (f00_11 * d0_00) + (f00_12 * d0_01) + (f00_21 * d0_10) + (f00_22 * d0_11) +
                    (f10_11 * d1_00) + (f10_12 * d1_01) + (f10_21 * d1_10) + (f10_22 * d1_11) +
                    (f20_11 * d2_00) + (f20_12 * d2_01) + (f20_21 * d2_10) + (f20_22 * d2_11) + b0;

  printf("expected pixel out: %6.4f\n", pixel_out);
}

void cpu_im2col_conv2D(ConvolDescriptor *D)
{
  _profile(_CPU_CONV2D, 0);
  //printf("CONV: input data : "); _profile_cpu_tensor(D->I);
  //printf("          filter : "); _profile_cpu_tensor(D->K);
  //printf("            bias : "); _profile_cpu_tensor(D->bias);

   int osize=D->z*D->r*D->c;
  int isize=D->r*D->c*D->kc*D->kr*D->kz;//r*c,kr*kc*kz

  float *ptrO=D->O->ptr;
  float *ptrI=D->ptrI;

  // obtenemos el ratio de similitud
  int batch_size   = D->I->shape[0];     // batch size
  float *I         = D->I->ptr;    // input activations
  int Irows        = D->I->shape[2];     // rows of input image
  int Icols        = D->I->shape[3];     // cols of input image
  int Ichannels    = D->I->shape[1];     // input channels
  unsigned int addr = 0;
  float v_ant;
  int eq = 0;
  int non_eq = 0;
  int eq_zero = 0;
  float max_dif = 0.f;
  float min = 9999.f;
  float max = -9999.f;
  #define MAX_SB 500000
  float sb_value[MAX_SB];
  int sb_valid[MAX_SB];
  int sb_num[MAX_SB];

// conv2D parameters
  int Krows        = D->kr;              // kernel rows
  int Kcols        = D->kc;              // kernel cols
  int Ochannels    = D->O->shape[1];     // output channels

  //cpu_print_data(D, Kcols, Krows, Ichannels, Ochannels, Icols, Irows);

  /*for (int n=0; n<MAX_SB; n++) {
    sb_valid[n] = 0;
  }
  
  for (int b=0;b<batch_size;b++) {
    for (int i=0; i<Ichannels; i++) {
      for (int r=0; r<Irows; r++) {
       for (int c=0; c<Icols; c++) {*/
          // score board
          /*int n;
          for (n=0; n<MAX_SB; n++) {
            if (sb_valid[n] == 0) break;
            if (sb_valid[n] && (sb_value[n] == I[addr])) {
              sb_num[n]++;
              break;
            }
          }
          if (n == MAX_SB) {
            printf("Error, too few SB entries\n"); exit(1);
          }
          if (!sb_valid[n]) {
            sb_valid[n] = 1;
            sb_value[n] = I[addr];
            sb_num[n] = 1;
            printf("allocating entry %d for value %f\n", n, I[addr]);
          }*/
          //
  /*        if (I[addr] == 0.f) eq_zero++;
          if (I[addr] < min) min = I[addr];
          if (I[addr] > max) max = I[addr];
          if (addr==0) {
            v_ant = I[addr];
            non_eq++;
            addr++;
          } else {
            if (fabs(v_ant - I[addr]) <= max_dif) {
              eq++;
            } else {
              non_eq++;
              //printf("%f %f\n", v_ant, I[addr]);
            }
            v_ant = I[addr];
            addr++;
          }
        }
      }
    }
  }
  int pixels = eq + non_eq;
  printf("\nSimilarity: equal to previous %9d, non_equal to previous %9d (%6.2f savings), zeroes %9d (%6.2f of total), min value %9f, max value %9f\n", eq, non_eq, 100.0f * (float)eq/(float)(pixels), eq_zero, 100.0f * (float)eq_zero/(float)(pixels), min, max);
  printf("Scoreboard: ");
  for (int n=0; n<MAX_SB; n++) {
    if (sb_valid[n] == 0) break;
    printf("value %f, num %d\n", sb_value[n], sb_num[n]);
  }*/
   
  // Map memory to Eigen
  Eigen::Map<Eigen::MatrixXf> matK=Eigen::Map<Eigen::MatrixXf>(D->K->ptr, D->kr * D->kc * D->kz, D->nk);

  #pragma omp parallel for
  for(int b=0;b<D->I->shape[0];b++){

    float *ptrO=D->O->ptr+(b*osize);
    float *ptrI=D->ptrI+(b*isize);

    Eigen::Map<Eigen::MatrixXf> matI=Eigen::Map<Eigen::MatrixXf>(ptrI,D->r*D->c,D->kz*D->kr*D->kc);
    Eigen::Map<Eigen::MatrixXf> matO=Eigen::Map<Eigen::MatrixXf>(ptrO,D->r*D->c,D->z);

    im2col(b,D,ptrI,0);

    matO=matI*matK;
  }// batch
    _profile(_CPU_CONV2D, 1);
}

void cpu_im2col_conv2D_grad(ConvolDescriptor *D)
{
  _profile(_CPU_CONV2D_GRAD, 0);
  //return;
  int osize=D->z*D->r*D->c;
  int isize=D->r*D->c*D->kc*D->kr*D->kz;//r*c,kr*kc*kz


  // Map memory to Eigen
  Eigen::Map<Eigen::MatrixXf> matgK=Eigen::Map<Eigen::MatrixXf>(D->gK->ptr, D->kr * D->kc * D->kz, D->nk);

  //#pragma omp parallel for
  for(int b=0;b<D->I->shape[0];b++){

    float *ptrD=D->D->ptr+(b*osize);
    float *ptrI=D->ptrI+(b*isize);

    Eigen::Map<Eigen::MatrixXf> matI=Eigen::Map<Eigen::MatrixXf>(ptrI,D->r*D->c,D->kz*D->kr*D->kc);
    Eigen::Map<Eigen::MatrixXf> matD=Eigen::Map<Eigen::MatrixXf>(ptrD,D->r*D->c,D->z);

    matgK+=matI.transpose()*matD;
  }// batch
    _profile(_CPU_CONV2D_GRAD, 1);
}

void cpu_im2col_conv2D_back(ConvolDescriptor *D)
{
  _profile(_CPU_CONV2D_BACK, 0);
  int osize=D->z*D->r*D->c;
  int isize=D->r*D->c*D->kc*D->kr*D->kz;//r*c,kr*kc*kz

  float *ptrD=D->D->ptr;
  float *ptrI=D->ptrI;

  // Map memory to Eigen
  Eigen::Map<Eigen::MatrixXf> matK=Eigen::Map<Eigen::MatrixXf>(D->K->ptr, D->kr * D->kc * D->kz, D->nk);

  #pragma omp parallel for
  for(int b=0;b<D->I->shape[0];b++){

    float *ptrD=D->D->ptr+(b*osize);
    float *ptrI=D->ptrI+(b*isize);

    Eigen::Map<Eigen::MatrixXf> matI=Eigen::Map<Eigen::MatrixXf>(ptrI,D->r*D->c,D->kz*D->kr*D->kc);
    Eigen::Map<Eigen::MatrixXf> matD=Eigen::Map<Eigen::MatrixXf>(ptrD,D->r*D->c,D->z);

    matI=matD*matK.transpose();

    im2col(b,D,ptrI,1);

  }// batch
    _profile(_CPU_CONV2D_BACK, 1);
}

void cpu_low_mem_conv3D(int batch_size,
        int channels, int image_depth, int image_rows, int image_cols, const float *image,
        int num_kernels, int kernel_depth, int kernel_rows, int kernel_cols, const float *kernel,
        int out_depth, int out_rows, int out_cols, float *output,
        int pad_depth, int pad_row, int pad_col,
        int stride_depth, int stride_rows, int stride_cols)
{
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++)
    for (int nk = 0; nk < num_kernels; nk++)
    for (int k = 0; k < out_depth; k++)
    for (int i = 0; i < out_rows; i++)
    for (int j = 0; j < out_cols; j++) {
        float s = 0;
        for (int z = 0; z < kernel_depth; z++) {
            int pz = k * stride_depth + z - pad_depth;
            if (pz >= 0 && pz < image_depth)
            for (int x = 0; x < kernel_rows; x++) {
                int px = i * stride_rows + x - pad_row;
                if (px >= 0 && px < image_rows)
                for (int y = 0; y < kernel_cols; y++) {
                    int py = j * stride_cols + y - pad_col;
                    if (py >= 0 && py < image_cols) {
                        for (int c = 0; c < channels; c++)
                            s += kernel[(((nk * channels + c) * kernel_depth + z) * kernel_rows + x) * kernel_cols + y]
                               * image[(((b * channels + c) * image_depth + pz) * image_rows + px) * image_cols + py];
                    }
                }
            }
        }
        output[(((b * num_kernels + nk) * out_depth + k) * out_rows + i) * out_cols + j] = s;
    }
}

void cpu_conv2D(ConvolDescriptor *D)
{

#ifdef CPU_DEBUG
        printf("conv2D:\n");
        printf(" input   : "); _profile_cpu_tensor(D->I);
        printf(" filters : "); _profile_cpu_tensor(D->K);
	if (D->use_bias) {printf(" bias    : "); _profile_cpu_tensor(D->bias);}
#endif	

  struct timeval time1, time2;
  gettimeofday(&time1, NULL);

    if (D->mem_level > 1) cpu_low_mem_conv3D(D->I->shape[0],
        D->iz, 1, D->ir, D->ic, D->I->ptr,
        D->nk, 1, D->kr, D->kc, D->K->ptr,
        1, D->r, D->c, D->O->ptr,
        0, D->padrt, D->padcl,
        1, D->sr, D->sc);
    else cpu_im2col_conv2D(D);

  int osize=D->z*D->r*D->c;
  //bias
  if (D->use_bias) {
    #pragma omp parallel for
    for(int b=0;b<D->O->shape[0];b++) {
      float *ptrO=D->O->ptr+(b*osize);
      for(int z=0;z<D->O->shape[1];z++)
      for(int r=0;r<D->O->shape[2];r++)
      for(int c=0;c<D->O->shape[3];c++,ptrO++)
      (*ptrO)+=D->bias->ptr[z];
    }
  }

    gettimeofday(&time2, NULL);
  unsigned long long t = ((time2.tv_sec - time1.tv_sec) * 1000000) + (time2.tv_usec - time1.tv_usec);
  #ifdef CPU_DEBUG
  printf("Conv2D: Time %llu us - %0dx%0dx%0dx%0d - KHxKW %0dx%0d\n", t, D->O->shape[1], D->I->shape[1], D->I->shape[2], D->I->shape[3], D->kr, D->kc);
  #endif


#ifdef CPU_DEBUG
  printf(" output  : "); _profile_cpu_tensor(D->O);
#endif
}

void cpu_low_mem_conv3D_grad(int batch_size,
        int channels, int image_depth, int image_rows, int image_cols, const float *image,
        int num_kernels, int kernel_depth, int kernel_rows, int kernel_cols, float *kernel,
        int out_depth, int out_rows, int out_cols, const float *delta,
        int pad_depth, int pad_row, int pad_col,
        int stride_depth, int stride_rows, int stride_cols)
{
    int kernel_size = num_kernels * channels * kernel_depth * kernel_rows * kernel_cols;
    for (int b = 0; b < batch_size; b++) {
        #pragma omp parallel for
        /* for (int nk = 0; nk < num_kernels; nk++)
        for (int c = 0; c < channels; c++)
        for (int z = 0; z < kernel_depth; z++)
        for (int x = 0; x < kernel_rows; x++)
        for (int y = 0; y < kernel_cols; y++) { */
        for (int tid = 0; tid < kernel_size; tid++) {
            int nk = tid;
            int y = nk % kernel_cols; nk /= kernel_cols;
            int x = nk % kernel_rows; nk /= kernel_rows;
            int z = nk % kernel_depth; nk /= kernel_depth;
            int c = nk % channels; nk /= channels;

            float s = 0.0;
            for (int k = 0; k < out_depth; k++) {
                int pz = k * stride_depth + z - pad_depth;
                if (pz < 0) continue;
                if (pz >= image_depth) continue;
                for (int i = 0; i < out_rows; i++) {
                    int px = i * stride_rows - pad_row + x;
                    if (px < 0) continue;
                    if (px >= image_rows) continue;
                    for (int j = 0; j < out_cols; j++) {
                        int py = j * stride_cols - pad_col + y;
                        if (py < 0) continue;
                        if (py >= image_cols) continue;
                        s += image[(((b * channels + c) * image_depth + pz) * image_rows + px) * image_cols + py] *
                            delta[(((b * num_kernels + nk) * out_depth + k) * out_rows + i) * out_cols + j];
                    }
                }
            }
            // kernel[((((nk * channels + c) * kernel_depth + z) * kernel_rows + x) * kernel_cols) + y] = s;
            kernel[tid] += s;
        }
    }
}

void cpu_conv2D_grad(ConvolDescriptor *D)
{
    if (D->mem_level > 1) cpu_low_mem_conv3D_grad(D->I->shape[0],
        D->iz, 1, D->ir, D->ic, D->I->ptr,
        D->nk, 1, D->kr, D->kc, D->gK->ptr,
        1, D->r, D->c, D->D->ptr,
        0, D->padrt, D->padcl,
        1, D->sr, D->sc);
    else cpu_im2col_conv2D_grad(D);

  //bias
  int osize=D->z*D->r*D->c;
  //#pragma omp parallel for
  if (D->use_bias) {
    for(int b=0;b<D->D->shape[0];b++) {
      float *ptrD=D->D->ptr+(b*osize);
      for(int z=0;z<D->D->shape[1];z++)
      for(int r=0;r<D->D->shape[2];r++)
      for(int c=0;c<D->D->shape[3];c++,ptrD++)
      D->gbias->ptr[z]+=(*ptrD);
    }
  }
}

void cpu_low_mem_conv3D_back(int batch_size,
        int channels, int image_depth, int image_rows, int image_cols, float *image,
        int num_kernels, int kernel_depth, int kernel_rows, int kernel_cols, const float *kernel,
        int out_depth, int out_rows, int out_cols, const float *delta,
        int pad_depth, int pad_row, int pad_col,
        int stride_depth, int stride_rows, int stride_cols)
{
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++)
    for (int c = 0; c < channels; c++)
    for (int k = 0; k < out_depth; k++)
    for (int i = 0; i < out_rows; i++)
    for (int j = 0; j < out_cols; j++)
        for (int z = 0; z < kernel_depth; z++) {
            int pz = k * stride_depth + z - pad_depth;
            if (pz < 0) continue;
            if (pz >= image_depth) continue;
            for (int x = 0; x < kernel_rows; x++) {
                int px = i * stride_rows - pad_row + x;
                if (px < 0) continue;
                if (px >= image_rows) continue;
                for (int y = 0; y < kernel_cols; y++) {
                    int py = j * stride_cols - pad_col + y;
                    if (py < 0) continue;
                    if (py >= image_cols) continue;
                    float s = 0.0;
                    for (int nk = 0; nk < num_kernels; nk++)
                        s += delta[(((b * num_kernels + nk) * out_depth + k) * out_rows + i) * out_cols + j]
                           * kernel[(((nk * channels + c) * kernel_depth + z) * kernel_rows + x) * kernel_cols + y];
                    image[(((b * channels + c) * image_depth + pz) * image_rows + px) * image_cols + py] += s;
                }
            }
        }
}

void cpu_conv2D_back(ConvolDescriptor *D)
{
    if (D->mem_level > 1) cpu_low_mem_conv3D_back(D->I->shape[0],
        D->iz, 1, D->ir, D->ic, D->ID->ptr,
        D->nk, 1, D->kr, D->kc, D->K->ptr,
        1, D->r, D->c, D->D->ptr,
        0, D->padrt, D->padcl,
        1, D->sr, D->sc);
    else cpu_im2col_conv2D_back(D);
}


void cpu_conv3D(ConvolDescriptor3D *D){
    cpu_low_mem_conv3D(D->I->shape[0],
        D->iz, D->id, D->ir, D->ic, D->I->ptr,
        D->nk, D->kd, D->kr, D->kc, D->K->ptr,
        D->d, D->r, D->c, D->O->ptr,
        D->paddf, D->padrt, D->padcl,
        D->sd, D->sr, D->sc);
}

void cpu_conv3D_grad(ConvolDescriptor3D *D){
    cpu_low_mem_conv3D_grad(D->I->shape[0],
        D->iz, D->id, D->ir, D->ic, D->I->ptr,
        D->nk, D->kd, D->kr, D->kc, D->gK->ptr,
        D->d, D->r, D->c, D->D->ptr,
        D->paddf, D->padrt, D->padcl,
        D->sd, D->sr, D->sc);
}

void cpu_conv3D_back(ConvolDescriptor3D *D){
    cpu_low_mem_conv3D_back(D->I->shape[0],
        D->iz, D->id, D->ir, D->ic, D->ID->ptr,
        D->nk, D->kd, D->kr, D->kc, D->K->ptr,
        D->d, D->r, D->c, D->D->ptr,
        D->paddf, D->padrt, D->padcl,
        D->sd, D->sr, D->sc);
}
