// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
//           Roberto Paredes Palacios, <rparedes@dsic.upv.es>
//           Jon Ander Gómez, <jon@dsic.upv.es>
//
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

#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <initializer_list>
#include <vector>
#include <string>
#include <iostream>

#include "tensor.h"
#include "utils.h"

#ifdef cGPU
#include "gpu/tensor_cuda.h"
#include "gpu/tensor_cuda_op.h"
#endif

using namespace std;

int initcuda[MAX_GPUS] = {0, 0, 0, 0, 0, 0, 0, 0};
int linpos;

extern ostream &operator<<(ostream &os, const tshape s);

void msg(string s, string s2) {
    cout << "\n" << s << " (" << s2 << ")\n";
    exit(0);
}


void msg(string s) { msg(s, ""); }

int Tensor::isCPU() { return (device == DEV_CPU); }

int Tensor::isGPU() { return ((device >= DEV_GPU) && (device < DEV_FPGA)); }

int Tensor::isFPGA() { return (device >= DEV_FPGA); }

// Tensor class
Tensor::Tensor() : device(DEV_CPU), ndim(0), size(0) {}

Tensor::Tensor(const initializer_list<int> &init) : Tensor(init, DEV_CPU) {}

Tensor::Tensor(const initializer_list<int> &init, int dev) : Tensor(tshape(init.begin(), init.end()), dev) {}

Tensor::Tensor(const tshape s) : Tensor(s, DEV_CPU) {}

Tensor::Tensor(tshape s, int dev) {
#ifndef cGPU
    if ((dev > DEV_CPU) && (isGPU())) {
        fprintf(stderr, "Not compiled for GPU\n");
        exit(0);
    }
#endif
#ifndef cFPGA
    if (dev >= DEV_FPGA) {
        fprintf(stderr, "Not compiled for FPGA\n");
        exit(0);
    }
#endif

    device = dev;
    ndim = s.size();
    shape = s;

    size = 1;
    for (int i = 0; i < ndim; ++i) size *= s[i];

    if (isCPU()) {
        if (ndim == 2) {
            mat = Eigen::MatrixXf(shape[1], shape[0]);
            ptr2 = &mat;
            ptr = &(mat(0, 0));
        } else {
            ptr = (float *) malloc(size * sizeof(float));
        }
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_device=device-DEV_GPU;
        if (!initcuda[gpu_device])
          {
            gpu_init(gpu_device);
            initcuda[gpu_device]=1;
          }
        ptr=gpu_create_tensor(gpu_device,size);
      }
#endif
#ifdef cFPGA
    else {
      // create FPGA Tensor
    }
#endif

    tsem = new mutex();
}


Tensor::Tensor(tshape s, Tensor *T) {
    device = T->device;
    ndim = s.size();
    shape = s;

    size = 1;
    for (int i = 0; i < ndim; ++i) size *= s[i];


    if (isCPU()) {
        ptr = T->ptr;
        if (ndim == 2) {
            new(&mat) Eigen::Map<Eigen::MatrixXf>(T->ptr, shape[1], shape[0]);
            ptr2 = &mat;
        }
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_device=device-DEV_GPU;
        ptr=T->ptr;
      }
#endif
#ifdef cFPGA
    else {
      // create FPGA Tensor
    }
#endif


    tsem = new mutex();
}

/////////////////////////////////////////////////////////////////////////
Tensor::Tensor(string fname, int bin) {
    FILE *fe;
    int i, j, v;

    if (bin) {
        fe = fopen(fname.c_str(), "rb");
        if (fe == NULL) {
            fprintf(stderr, "%s not found\n", fname.c_str());
            exit(1);
        }

        int read = fread(&ndim, sizeof(int), 1, fe);
        for (int i = 0; i < ndim; ++i) {
            int read = fread(&v, sizeof(int), 1, fe);
            shape.push_back(v);
        }

        tshape s = shape;

        cout << "loading file with tensor:" << s << "\n";

        device = DEV_CPU;
        size = 1;
        for (int i = 0; i < ndim; ++i) size *= shape[i];

        if (ndim == 2) {
            //ptr=(float *)malloc(size*sizeof(float));
            //Eigen::Map<Eigen::MatrixXf> mat(ptr,shape[1],shape[0]);
            //ptr2=&mat;

            mat = Eigen::MatrixXf(shape[1], shape[0]);
            ptr2 = &mat;
            ptr = &(mat(0, 0));

        } else {
            ptr = (float *) malloc(size * sizeof(Tensor *));
        }

        tsem = new mutex();

        read = fread(ptr, sizeof(float), size, fe);
        if (read != size) {
            fprintf(stderr, "Error reading file (%d!=%d)\nCheck format\n", read, size);
            exit(1);
        }

        fclose(fe);
    } else {
        fe = fopen(fname.c_str(), "rt");
        if (fe == NULL) {
            fprintf(stderr, "%s not found\n", fname.c_str());
            exit(1);
        }

        fscanf(fe, "%d ", &ndim);
        for (int i = 0; i < ndim; ++i) {
            fscanf(fe, "%d ", &v);
            shape.push_back(v);
        }

        tshape s = shape;

        cout << "loading file with tensor:" << s << "\n";

        device = DEV_CPU;
        size = 1;
        for (int i = 0; i < ndim; ++i) size *= shape[i];

        if (ndim == 2) {
            mat = Eigen::MatrixXf(shape[1], shape[0]);
            ptr2 = &mat;
            ptr = &(mat(0, 0));
        } else {
            ptr = (float *) malloc(size * sizeof(Tensor *));
        }

        tsem = new mutex();

        for (int i = 0; i < size; i++) fscanf(fe, "%f ", &(ptr[i]));

        fclose(fe);
    }
}


///////////////////////////////////////////
void Tensor::save(string fname) {
    if (!isCPU())
        msg("Only save CPU Tensors", "Tensor::save");

    int i, j;
    FILE *fe;
    float fv;

    fe = fopen(fname.c_str(), "wb");
    if (fe == NULL) {
        fprintf(stderr, "Not abel to write %s \n", fname.c_str());
        exit(1);
    }

    fprintf(stderr, "writting bin file\n");

    fwrite(&ndim, sizeof(int), 1, fe);
    for (i = 0; i < ndim; ++i)
        fwrite(&shape[i], sizeof(int), 1, fe);

    fwrite(ptr, sizeof(float), size, fe);

    fclose(fe);

}


///////////////////////////////////////////
Tensor *Tensor::share() {
    Tensor *C = new Tensor(getshape(), device);

    return C;

}


///////////////////////////////////////////
Tensor::~Tensor() {
    if (isCPU()) {
        if (ndim != 2) free(ptr);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_delete_tensor(gpu_device,ptr);
      }
#endif
#ifdef cFPGA
    else {
      // delete FPGA Tensor
    }
#endif
    delete tsem;
}


///////////////////////////////////////////
tshape Tensor::getshape() {
    tshape s = shape;
    return s;
}


///////////////////////////////////////////
void Tensor::info() {
    int i;

    fprintf(stderr, "DIM=%d\n", ndim);
    fprintf(stderr, "(");
    for (i = 0; i < ndim - 1; i++)
        fprintf(stderr, "%d,", shape[i]);
    fprintf(stderr, "%d)\n", shape[i]);

    fprintf(stderr, "Total bytes=%ld\n", size * sizeof(float));
    if (isCPU()) fprintf(stderr, "Device=CPU\n");
    else if (isGPU()) fprintf(stderr, "Device=GPU (%d)\n", gpu_device);
    else fprintf(stderr, "Device=FPGA\n");
}




///////////////////////////////////////////

void Tensor::print() {

    if (isCPU()) {
        if (ndim == 1)
            for (int i = 0; i < shape[0]; ++i)
                printf("%f ", ptr[i]);
        else if (ndim == 2) {
            cout << (*ptr2).transpose() << "\n";
        } else {
            int i;
            for (i = 0; i < size; ++i)
                printf("%f ", ptr[i]);
            printf("\n");
        }
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_set_device(gpu_device);
        float *v= (float*)malloc(size*sizeof(float));
        cudaMemcpy(v,ptr,size*sizeof(float),cudaMemcpyDeviceToHost);
        if (ndim==2)
          {
            int i,j,p=0;
            for(i=0;i<shape[0];++i)
              {
                for(j=0;j<shape[1];++j,++p)
                  printf("%f ",v[p]);
                  printf("\n");
              }
          }
        else
          {
            int i;
            for(i=0;i<size;++i)
              printf("%f ",v[i]);
              printf("\n");
          }
          free(v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    cout << "\n";
}


///////////////////////////////////////////
void Tensor::set(float v) {
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = v;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_set(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


///////////////////////////////////////////
void Tensor::mult(float v) {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] *= v;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_mult(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


///////////////////////////////////////////
void Tensor::div(float v) { mult(1.0 / v); }

///////////////////////////////////////////
void Tensor::sum(float v) {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] += v;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_sum(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

///////////////////////////////////////////
void Tensor::sub(float v) { sum(-v); }

///////////////////////////////////////////
void Tensor::set_log() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = log(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_log(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

///////////////////////////////////////////
void Tensor::set_exp() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = exp(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_exp(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

///////////////////////////////////////////
void Tensor::set_sqrt() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = sqrt(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_sqrt(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

///////////////////////////////////////////
void Tensor::set_sqr() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] *= ptr[i];
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_sqr(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


///////////////////////////////////////
float Tensor::total_sum() {

    if (isCPU()) {
        float sum = 0.0;

        for (int i = 0; i < size; ++i) sum += ptr[i];

        return sum;
    }
#ifdef cGPU
    else if (isGPU())
      {
         float sum;
         gpu_total_sum(this,&sum);
         return sum;
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    return 0;
}

//////////////////////////////////////
float Tensor::total_abs() {

    if (isCPU()) {
        float sum = 0.0;

        for (int i = 0; i < size; ++i) sum += fabs(ptr[i]);

        return sum;
    }
#ifdef cGPU
    else if (isGPU())
      {
         float sum;
         gpu_total_sum(this,&sum);
         return sum;
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    return 0;
}


///////////////////////////////////////////
void Tensor::rand_uniform(float v) {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = uniform() * v;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_uniform(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

}


///////////////////////////////////////////
void Tensor::rand_suniform(float v) {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = suniform() * v;

    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_suniform(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif


}


///////////////////////////////////////////
void Tensor::rand_gaussian(float m, float s) {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = gauss(m, s);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_gaussian(this,m,s);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

}


void Tensor::rand_binary(float v) {
    if (isCPU()) {
        for (int i = 0; i < size; ++i)
            if (uniform() < v) ptr[i] = 1.0;
            else ptr[i] = 0.0;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_binary(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

}


///////////////////////////////////////////
