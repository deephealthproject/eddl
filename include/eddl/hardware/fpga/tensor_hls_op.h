// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019 Roberto Paredes Palacios, <rparedes@dsic.upv.es>

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef _TENSOR_HLS_OP_
#define _TENSOR_HLS_OP_

#include "xcl2.hpp"
#include "eddl/tensor/tensor.h"

#define FPGAMULT 0
#define FPGASUM 1
#define FPGALOG 2
#define FPGAEXP 3
#define FPGASQRT 4
#define FPGASQR 5
#define FPGATOTALSUM 6
#define FPGATOTALABS 7
#define FPGASET 8
#define FPGARELU 9
#define FPGASOFTM 10
#define FPGAGAUSS 11
#define FPGAFILL_ 20

int load_file_to_memory(const char *filename, char **result);
void tensor_op_hls(Tensor *A, float fp, int kernel_id);
void fpga_init();
void fpga_create_tensor(Tensor *T, int dev);
void fpga_delete_tensor(Tensor *T);
void fpga_tensor_operation(Tensor *A, Tensor *B, int kernel_id);
void fpga_copy_fpga(Tensor *A, Tensor *B);
void fpga_copy_to_fpga(float *nptr,Tensor *A);
void fpga_copy_from_fpga(Tensor *A,float *nptr);
void fpga_tensor_add(float scA,Tensor *A, float scB,Tensor *B, Tensor *C,int incC);
void fpga_mult2D(Tensor *A,int tA, Tensor *B, int tB, Tensor *C, int incC);
void fpga_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
void fpga_cent(Tensor *A, Tensor *B, Tensor *C);
void fpga_relu_soft_d(Tensor *D, Tensor *I, Tensor *PD, int kernel_id);
void fpga_reduce_sum2D(Tensor *A, Tensor *B, int axis,int incB);
int fpga_accuracy (Tensor *A, Tensor *B);
float fpga_total_sum (Tensor *A);
void fpga_el_div_mult(Tensor *A, Tensor *B, Tensor *C, int incC, int op);
void fpga_tensor_normalize(Tensor *A, float max, float min);
void fpga_gemx_mult2D(Tensor *A,int tA, Tensor *B, int tB, Tensor *C, int incC);
void fpga_gemx_mult2D_CPU(Tensor *A,int tA, Tensor *B, int tB, Tensor *C, int incC);
void verify(Tensor *T);
void verify2(cl::Buffer &buffer, int tam);
//int load_file_to_memory(const char *filename, char **result);
//void tensor_op_hls(void *A, int tam, float fp, int kernel_id);
//void fpga_init();
//void* fpga_create_tensor(int dev,int size);
//void tensor_operations(void* A, void* B, int tam, int kernel_id);
void fpga_core(Tensor *A, float v, int kernel_id);



#endif
