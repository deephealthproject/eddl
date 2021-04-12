/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>

#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"

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

void cpu_im2col_conv2D(ConvolDescriptor *D)
{
  _profile(_CPU_CONV2D, 0);
  int osize=D->z*D->r*D->c;
  int isize=D->r*D->c*D->kc*D->kr*D->kz;//r*c,kr*kc*kz

  float *ptrO=D->O->ptr;
  float *ptrI=D->ptrI;


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

void cpu_new_im2col(int b, ConvolDescriptor *D, float *ptrI)
{
    for (int r = 0; r < D->r; r++)
        for (int c = 0; c < D->c; c++)
            for (int kz = 0; kz < D->kz; kz++)
                for (int kr = 0; kr < D->kr; kr++)
                    for (int kc = 0; kc < D->kc; kc++) {
                        int i = r * D->c + c;
                        int j = ((kz * D->kr + kr) * D->kc) + kc;
                        int k = j * D->r * D->c + i;
                        int y = r * D->sr - D->padrt + kr;
                        int x = c * D->sc - D->padcl + kc;
                        // ptrI[k]=get_pixel(b,x,y,pz,D,isize,irsize);
                        // float get_pixel(int b,int px,int py,int pz,ConvolDescriptor *D,int isize,int irsize)
                        // Check boundaries of the window
                        if (x<0 || y<0 || x>=D->ic || y>=D->ir) ptrI[k] = 0.0;
                        else {
                            // Compute address from indices (row-major)
                            unsigned int address = b*D->iz*D->ir*D->ic + kz*D->ir*D->ic + y*D->ic + x;
                            ptrI[k] = D->I->ptr[address];
                        }
                    }
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
    for (int c = 0; c < channels; c++)
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
                        s += kernel[(((nk * channels + c) * kernel_depth + z) * kernel_rows + x) * kernel_cols + y]
                           * image[(((b * channels + c) * image_depth + pz) * image_rows + px) * image_cols + py];
                    }
                }
            }
        }
        output[(((b * num_kernels + nk) * out_depth + k) * out_rows + i) * out_cols + j] += s;
    }
}

void cpu_conv2D(ConvolDescriptor *D)
{
#if 0
    // cpu_im2col_conv2D(D); // Conv2D_grad depends on im2col stored in ptrI
    cpu_low_mem_conv3D(D->I->shape[0],
        D->iz, 1, D->ir, D->ic, D->I->ptr,
        D->nk, 1, D->kr, D->kc, D->K->ptr,
        1, D->r, D->c, D->O->ptr,
        0, D->padrt, D->padcl,
        1, D->sr, D->sc);
#else
    // printf("input: shape[0]=%d, in=%d\n", D->I->shape[0], D->in);
    /* printf("input: in=%d, iz=%d, ir=%d, ic=%d\n", D->I->shape[0], D->iz, D->ir, D->ic);
    printf("kernel: nk=%d, kz (iz)=%d, kr=%d, kc=%d\n", D->nk, D->kz, D->kr, D->kc);
    printf("output: in=%d, z (nk)=%d, r=%d, c=%d\n", D->in, D->z, D->r, D->c);
    printf("padcl=%d, padcr=%d, padrt=%d, padrb=%d\n", D->padcl, D->padcr, D->padrt, D->padrb);
    printf("stride: %d,%d bias:%d\n", D->sr, D->sc, D->use_bias); */
    cpu_im2col_conv2D(D);
    int n = D->I->shape[0] * D->z *D->r * D->c;
    float *output = new float[n];
    memset(output, 0, n * sizeof(float));
    cpu_low_mem_conv3D(D->I->shape[0],
        D->iz, 1, D->ir, D->ic, D->I->ptr,
        D->nk, 1, D->kr, D->kc, D->K->ptr,
        1, D->r, D->c, output,
        0, D->padrt, D->padcl,
        1, D->sr, D->sc);
    int pos = 0; float max = 0.0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(output[i] - D->O->ptr[i]);
        if (fabs(D->O->ptr[i]) > 1e-7) d = d / fabsf(D->O->ptr[i]);
        if (d > max) { max = d; pos = i; }
    }
    printf("cpu_conv2D     : %e (%e,%e)\n", max, output[pos], D->O->ptr[pos]);
    delete output;
#endif
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
}

void cpu_low_mem_conv2D_grad(int batch_size,
        int channels, int image_rows, int image_cols, const float *image,
        int num_kernels, int kernel_rows, int kernel_cols, float *kernel,
        int out_rows, int out_cols, const float *delta,
        int pad_row, int pad_col,
        int stride_rows, int stride_cols)
{
    // #pragma omp parallel for
    for (int b = 0; b < batch_size; b++)
    for (int c = 0; c < channels; c++)
    for (int nk = 0; nk < num_kernels; nk++)
    for (int i = 0; i < out_rows; i++)
    for (int j = 0; j < out_cols; j++)
    for (int x = 0; x < kernel_rows; x++) {
        int px = i * stride_rows - pad_row + x;
        if (px < 0) continue;
        if (px >= image_rows) continue;
        for (int y = 0; y < kernel_cols; y++) {
            int py = j * stride_cols - pad_col + y;
            if (py < 0) continue;
            if (py >= image_cols) continue;
            kernel[(((nk * channels + c) * kernel_rows + x) * kernel_cols) + y] +=
               image[((b * channels + c) * image_rows + px) * image_cols + py] *
               delta[((b * num_kernels + nk) * out_rows + i) * out_cols + j];
        }
    }
}

void cpu_conv2D_grad(ConvolDescriptor *D)
{
#if 0
    // cpu_im2col_conv2D_grad(D);
    cpu_low_mem_conv2D_grad(D->I->shape[0],
        D->iz, D->ir, D->ic, D->I->ptr,
        D->nk, D->kr, D->kc, D->gK->ptr,
        D->r, D->c, D->D->ptr,
        D->padrt, D->padcl,
        D->sr, D->sc);
#else
    cpu_im2col_conv2D_grad(D);
    int n = D->kr * D->kc * D->kz * D->nk;
    float *output = new float[n];
    memset(output, 0, D->kr * D->kc * D->kz * D->nk * sizeof(float));
    cpu_low_mem_conv2D_grad(D->I->shape[0],
        D->iz, D->ir, D->ic, D->I->ptr,
        D->nk, D->kr, D->kc, output,
        D->r, D->c, D->D->ptr,
        D->padrt, D->padcl,
        D->sr, D->sc);
    int pos = 0; float max = 0.0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(output[i] - D->gK->ptr[i]);
        if (fabs(D->gK->ptr[i]) > 1e-7) d = d / fabsf(D->gK->ptr[i]);
        if (d > max) { max = d; pos = i; }
    }
    printf("cpu_conv2D_grad: %e (%e,%e)\n", max, output[pos], D->gK->ptr[pos]);
    delete output;
#endif
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

void cpu_low_mem_conv2D_back(int batch_size,
        int channels, int image_rows, int image_cols, float *image,
        int num_kernels, int kernel_rows, int kernel_cols, const float *kernel,
        int out_rows, int out_cols, const float *delta,
        int pad_row, int pad_col,
        int stride_rows, int stride_cols)
{
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++)
    for (int c = 0; c < channels; c++)
    for (int i = 0; i < out_rows; i++)
    for (int j = 0; j < out_cols; j++)
        for (int x = 0; x < kernel_rows; x++) {
            int px = i * stride_rows - pad_row + x;
            if (px < 0) continue;
            if (px >= image_rows) continue;
            for (int y = 0; y < kernel_cols; y++) {
                int py = j * stride_cols - pad_col + y;
                if (py < 0) continue;
                if (py >= image_cols) continue;
                double a = 0.0;
                for (int nk = 0; nk < num_kernels; nk++)
                    a += delta[((b * num_kernels + nk) * out_rows + i) * out_cols + j]
                       * kernel[((nk * channels + c) * kernel_rows + x) * kernel_cols + y];
                image[((b * channels + c) * image_rows + px) * image_cols + py] += a;
            }
        }
}

void cpu_conv2D_back(ConvolDescriptor *D)
{
#if 0
    // cpu_im2col_conv2D_back(D);
    cpu_low_mem_conv2D_back(D->I->shape[0],
        D->iz, D->ir, D->ic, D->ID->ptr,
        D->nk, D->kr, D->kc, D->K->ptr,
        D->r, D->c, D->D->ptr,
        D->padrt, D->padcl,
        D->sr, D->sc);
#else
    cpu_im2col_conv2D_back(D);
    int n = D->I->shape[0] * D->iz * D->ir * D->ic;
    float *output = new float[n];
    memset(output, 0, n * sizeof(float));
    cpu_low_mem_conv2D_back(D->I->shape[0],
        D->iz, D->ir, D->ic, output,
        D->nk, D->kr, D->kc, D->K->ptr,
        D->r, D->c, D->D->ptr,
        D->padrt, D->padcl,
        D->sr, D->sc);
    int pos = 0; float max = 0.0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(output[i] - D->ID->ptr[i]);
        if (fabs(D->ID->ptr[i]) > 1e-7) d = d / fabsf(D->ID->ptr[i]);
        if (d > max) { max = d; pos = i; }
    }
    printf("cpu_conv2D_back: %e (%e,%e)\n", max, output[pos], D->ID->ptr[pos]);
    delete output;
#endif
}


void cpu_conv3D(ConvolDescriptor3D *D){

}

void cpu_conv3D_grad(ConvolDescriptor3D *D){

}

void cpu_conv3D_back(ConvolDescriptor3D *D){

}
