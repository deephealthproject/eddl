/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_TENSOR_H
#define EDDL_TENSOR_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <string>
#include <mutex>

#include "Eigen/Dense"

#include "eddl/utils.h"
#include "eddl/descriptors/tensor_descriptors.h"

// Read/Write Numpy
//#include "eddl/tensor/cnpy/cnpy.h"

#define DEV_CPU 0

#define DEV_GPU 1000
#define DEV_GPU_0 1000
#define DEV_GPU_1 1001
#define DEV_GPU_2 1002
#define DEV_GPU_3 1003
#define DEV_GPU_4 1004
#define DEV_GPU_5 1005
#define DEV_GPU_6 1006
#define DEV_GPU_7 1007
#define DEV_GPU_8 1008

#define DEV_FPGA 2000
#define DEV_FPGA_0 2000
#define DEV_FPGA_1 2001
#define DEV_FPGA_2 2002
#define DEV_FPGA_3 2003
#define DEV_FPGA_4 2004
#define DEV_FPGA_5 2005
#define DEV_FPGA_6 2006
#define DEV_FPGA_7 2007
#define DEV_FPGA_8 2008

#define MAX_GPUS 8

using namespace std;

// TODO: Remove this. Don't like here
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> MatrixXRMf;
typedef vector<int> tshape;

class Tensor {
private:
    // Load methods
    static Tensor* load_from_bin(std::ifstream &ifs);
    static Tensor* load_from_onnx(std::ifstream &ifs);
    static Tensor* load_from_img(const string &filename, const string &format);
//    template<typename T> static Tensor* load_from_numpy(const string &filename, const string &format);  // Deprecated
//    static Tensor* load_from_txt(std::ifstream &ifs, char delimiter, int headerRows);  // Deprecated

    // Save methods
    void save2bin(std::ofstream &ofs);
    void save2onnx(std::ofstream &ofs);
    void save2img(const string &filename, string format);
//    void save2numpy(const string &filename, string format);
    void save2txt(std::ofstream &ofs, const char delimiter, const vector<string> &header);

public:
    int device;
    unsigned int ndim;
    unsigned long int size;
    vector<int> shape;
    vector<int> stride;

    // Data pointers
    float *ptr;
    Eigen::MatrixXf *ptr2;  // TODO: I don't like it. float or eigen, not both

    // Aux variables
    int gpu_device;
    mutex *tsem;  // Multithreading. Tensor semaphore

    // Constructors
    Tensor();
    explicit Tensor(const vector<int> &shape, int dev=DEV_CPU);
    Tensor(const vector<int> &shape, float *fptr, int dev);
    Tensor(const vector<int> &shape, Tensor *T);
    Tensor(const vector<float>& data, const vector<int> &shape, int dev=DEV_CPU);

    // Destructors
    ~Tensor();

    // Internal methods
    void updateDevice(int dev);
    void updateShape(const vector<int> &new_shape);
    void updateSize();
    void updateStrides();
    void updateData(float* ptr);
    void deleteData();

    /**
      *  @brief Clone a tensor to the CPU.
    */
    void toCPU(int dev=DEV_CPU);

    /**
      *  @brief Clone a tensor to the GPU.
    */
    void toGPU(int dev=DEV_GPU);

    /**
      *  @brief Check if the tensor is in CPU.
      *
      *  @return int
    */
    int isCPU();

    /**
      *  @brief Check if the tensor is in GPU.
      *
      *  @return int
    */
    int isGPU();

    /**
      *  @brief Check if the tensor is in FPGA.
      *
      *  @return int
    */
    int isFPGA();


    /**
      *  @brief Print shape, device and size information.
      *
      *  @return    void
    */
    void info();

    /**
      *  @brief Print the tensor values.
      *
      *  @return    void
    */
    void print(int precision=6, bool raw=false);

    /**
      *  @brief Returns the device name where the tensor is allocated ("CPU", "GPU" or "FPGA")
      *
      *  @return    string
    */
    string getDeviceName();

    // Core
    vector<int> getShape();
    unsigned int numel();

    /**
      *  @brief Check if all dimensions in the tensor are the same.
      *
      *  @param A   Tensor
      *  @return    bool
    */
    static bool isSquared(Tensor *A);


    // Serialization *****************************
    /**
      *  @brief Load tensor from filestream.
      *
      *  @param ifs  Filestream
      *  @param format    File format. Accepted formats are: bin, onnx, csv, tsv, txt.
      *  @return    Tensor
    */
    static Tensor* loadfs(std::ifstream &ifs, string format="");

    /**
      *  @brief Load tensor from file.
      *
      *  @param filename  Name of the file to load the tensor from.
      *  @param format    Filetype. The accepted filetypes are the following:
      *                     - Images: jpg, jpeg, png, bmp, hdr, psd, tga, gif, pic, pgm, ppm.
      *                     - Numpy: npy, npz
      *                     - Other: bin, onnx
      *  @return    Tensor
    */
    static Tensor* load(const string& filename, string format="");
    template<typename T> static Tensor* load(const string& filename, string format="");

//    /**
//      *  @brief Load data from a text file
//      *
//      *  @param filename  Name of the file to load the tensor from.
//      *  @param delimiter    Character used to separate the columns of the file.
//      *  @param headerRows   Number of top rows to avoid, generally because they correspond to the header.
//      *  @return    Tensor
//    */
//    static Tensor* load_from_txt(const string& filename, const char delimiter=',', int headerRows=1);

    /**
      *  @brief Save tensor to a filestream.
      *
      *  @param ofs     Filestream.
      *  @param format    Format to use. The accepted formats are the following:
      *                     - Text: csv, tsv, txt
      *                     - Other: bin, onnx
      *  @return    void
    */
    void savefs(std::ofstream &ofs, string format="");

    /**
      *  @brief Save tensor to a file.
      *
      *  @param filename    Name of the file to save the tensor to.
      *  @param format    Filetype. The accepted filetypes are the following:
      *                     - Images: png, bmp, tga, jpg, jpeg, hdr.
      *                     - Numpy: npy, npz
      *                     - Text: csv, tsv, txt
      *                     - Other: bin, onnx
      *  @return    void
    */
    void save(const string& filename, string format="");

    /**
      *  @brief Save tensor to a text file.
      *
      *  @param filename    Name of the file to save the tensor to.
      *  @param delimiter   Character to use to separate the columns of the file.
      *  @param header      Header rows.
      *  @return    void
    */
    void save2txt(const string& filename, const char delimiter=',', const vector<string> &header={});

    // ************************************************
    // ****** Tensor operations ***********************
    // ************************************************

    // Creation ops ***********************************

    /**
      *  @brief Create a tensor of the specified shape filled with uninitialized data
      *
      *  @param shape  Shape of the tensor to create.
      *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
      *  @return     Tensor of the specified shape filled with zeros
     */
    static Tensor* empty(const vector<int> &shape, int dev=DEV_CPU);

    /**
      *  @brief Create a tensor with the shape and device of another one, but empty
      *
      *  @param A  Input tensor from wich to take shape and device.
      *  @return     Empty initialized A-shaped tensor 
     */
    static Tensor* empty_like(Tensor *A);


    /**
      *  @brief Create a tensor of the specified shape and filled with zeros.
      *
      *  @param shape  Shape of the tensor to create.
      *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
      *  @return     Tensor of the specified shape filled with zeros
    */
    static Tensor* zeros(const vector<int> &shape, int dev=DEV_CPU);

    /**
      *  @brief Create a tensor with the shape and device of another one, initialized with zeros
      *
      *  @param A  Input tensor from wich to take shape and device.
      *  @return     Zeros initialized A-shaped tensor 
     */
    static Tensor* zeros_like(Tensor *A);

    /**
      *  @brief Create a tensor of the specified shape and filled with ones.
      *
      *  @param shape  Shape of the tensor to create.
      *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
      *  @return     Tensor of the specified shape filled with ones
    */
    static Tensor* ones(const vector<int> &shape, int dev=DEV_CPU);

    /**
      *  @brief Create a tensor with the shape and device of another one, initialized with ones
      *
      *  @param A  Input tensor from wich to take shape and device.
      *  @return     Ones initialized A-shaped tensor 
     */
    static Tensor* ones_like(Tensor *A);

    /**
      *  @brief Create a tensor of the specified shape and filled with a specific value.
      *
      *  @param shape  Shape of the tensor to create.
      *  @param value  Value to use to fill the tensor.
      *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
      *  @return     Tensor of the specified shape filled with the value
    */
    static Tensor* full(const vector<int> &shape, float value, int dev=DEV_CPU);

    /**
      *  @brief Create a tensor with the shape an device of the input tensor and filled with a specific value.
      *
      *  @param A  Input tensor from wich to take shape and device.
      *  @return     Value initialized A-shaped tensor 
    */
    static Tensor* full_like(Tensor *A, float value);

    /**
    *   @brief Create a 1-D tensor of size ceil(end - start) with values from start to end with step step.
    *
    *   @param start Start index
    *   @param end  End index
    *   @param step  The gap between two values in the tensor.
    *   @param dev One of ``DEV_CPU``or ``DEV_GPU``
    *   @return The new tensor
    */
    static Tensor* arange(float start, float end, float step=1.0f, int dev=DEV_CPU);

    /**
    *   @brief Creates a 1-D tensor of size floor(end - start)/ + 1 with values from start to end with step step.
    *
    *   @param start Start value
    *   @param end  End value
    *   @param step  The gap between two values in the tensor.
    *   @param dev One of ``DEV_CPU``or ``DEV_GPU``
    *   @return The new tensor
    */
    static Tensor* range(float start, float end, float step=1.0f, int dev=DEV_CPU);

    /**
    *   @brief Creates a 1-D tensor with a sequence of num evenly-spaced values starting at start. If steps > 1, the values in the sequence increase by end - start / steps - 1, so that the last one is exactly end.
    *   @param start Start value
    *   @param end  End value
    *   @param steps  The gap between two values in the tensor.
    *   @param dev One of ``DEV_CPU``or ``DEV_GPU``
    *   @return The new tensor
    */
    static Tensor* linspace(float start, float end, int steps=100, int dev=DEV_CPU);


    /**
    *   @brief Creates a 1-D tensor with a sequence of num  logarithmic spaced values starting at start. If steps > 1, the values in the sequence increase by end - start / steps - 1, so that the last one is exactly end.
    *   @param start Start value
    *   @param end  End value
    *   @param steps  The gap between two values in the tensor.
    *   @param base  The base of the logarithm to apply.
    *   @param dev One of ``DEV_CPU``or ``DEV_GPU``
    *   @return The new tensor
    */
    static Tensor* logspace(float start, float end, int steps=100, float base=10.0f, int dev=DEV_CPU);

    /**
    *   @brief Creates a 1-D tensor with a sequence of num  geometrically spaced values starting at start. If steps > 1, the values in the sequence increase by end - start / steps - 1, so that the last one is exactly end.
    *   @param start Start value
    *   @param end  End value
    *   @param steps  The gap between two values in the tensor.
    *   @param dev One of ``DEV_CPU``or ``DEV_GPU``
    *   @return The new tensor
    */
    static Tensor* geomspace(float start, float end, int steps=100, int dev=DEV_CPU);

    /**
      *  @brief
      *
      *  @param rows  Number of rows of the tensor.
      *  @param offset
      *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
      *  @return     Tensor of the specified shape filled with the value
    */
    static Tensor* eye(int rows, int offset=0, int dev=DEV_CPU);

    /**
      *  @brief Create a tensor representing the identity matrix. Equivalent to calling function ``eye`` with ``offset = 0``.
      *
      *  @param shape  Shape of the tensor to create.
      *  @param value  Value to use to fill the tensor.
      *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
      *  @return     Tensor of the specified shape filled with the value
    */
    static Tensor* identity(int rows, int dev=DEV_CPU);

    /**
      *  @brief Initialise a tensor with random values following an uniform distribution.
      *
      *  @param shape  Shape of the tensor to create.
      *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
      *  @return     Tensor of the specified shape filled
    */
    static Tensor* randu(const vector<int> &shape, int dev=DEV_CPU);

    /**
      *  @brief Initialise a tensor with random values following an normal distribution.
      *
      *  @param shape  Shape of the tensor to create.
      *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
      *  @return     Tensor of the specified shape filled
    */
    static Tensor* randn(const vector<int> &shape, int dev=DEV_CPU);

    /**
      *  @brief Obtain the elements in a squared matrix diagonal. Inplace operation.
      *
      *  @param k  Offset. If k=0, main diagonal is selected. If k>0, a diagonal above the main diagonal is selected. If k<0, a diagonal below the main diagonal is selected.
    */
    void diag_(int k=0);

    /**
      *  @brief Obtain the elements in a squared matrix diagonal.
      *
      *  @param k  Offset. If k=0, main diagonal is selected. If k>0, a diagonal above the main diagonal is selected. If k<0, a diagonal below the main diagonal is selected.
      *  @return  A new tensor with the elements on the selected diagonal.
    */
    Tensor* diag(int k=0);

    /**
      *  @brief Obtain the elements in a squared matrix diagonal.
      *
      *  @param A  Input matrix.
      *  @param B  Output matrix.
      *  @param k  Offset. If k=0, main diagonal is selected. If k>0, a diagonal above the main diagonal is selected. If k<0, a diagonal below the main diagonal is selected.
    */
    static void diag(Tensor* A, Tensor* B, int k=0);

    // Math operations (Tensor-Tensor, Tensor-float) ************************

    /**
      *  @brief Apply a lower bound to the elements in a tensor.
      *
      *  @param v  Lower bound.
      *  @return A new tensor with the values lower than v set to v.
    */
    Tensor* maximum(float v);

    /**
      *  @brief Apply a lower bound to the elements in a tensor.
      *
      *  @param A  Input tensor.
      *  @param v  Lower bound.
      *  @return A new tensor with the values of A lower than v set to v.
    */
    static Tensor* maximum(Tensor* A, float v);

    /**
      *  @brief Apply a lower bound to the elements in a tensor.
      *
      *  @param A  Input tensor.
      *  @param B  Output tensor.
      *  @param v  Lower bound.
    */
    static void maximum(Tensor* A, Tensor* B, float v);

    /**
      *  @brief Element-wise selection of the maximum values in the same position in two tensors.
      *
      *  @param A  Input tensor.
      *  @param B  Input tensor.
      *  @return  A tensor with the higher value in the same position between A and B.
    */
    static Tensor* maximum(Tensor* A, Tensor* B);

    /**
      *  @brief Element-wise selection of the maximum values in the same position in two tensors.
      *
      *  @param A  Input tensor.
      *  @param B  Input tensor.
      *  @param C  Output tensor with the higher value in the same position between A and B.
    */
    static void maximum(Tensor* A, Tensor* B, Tensor* C);

   
    Tensor* minimum(float v);
    static Tensor* minimum(Tensor* A, float v);
    static void minimum(Tensor* A, Tensor* B, float v);

    static Tensor* minimum(Tensor* A, Tensor* B);
    static void minimum(Tensor* A, Tensor* B, Tensor* C);

    // Math operations (single-value) ************************

    /**
      *  @brief Get the median value of all the elements in the tensor.
      *
      *  @return The median value of all the elements in the tensor.
    */
    float median();

    /**
      *  @brief Get the median value of all the elements in the input tensor.
      *
      *  @param A Input tensor.
      *  @return The median value of all the elements in the input tensor.
    */
    static float median(Tensor* A);

    // Math operations (reductions) ************************

    /**
    *   @brief Obtain the maximum value in the tensor
    *   @return float. The maximum value in the tensor
    */
    float max();

    /**
    *   @brief Obtain the maximum value in a tensor
    *   @param A The tensor where the operation is applied
    *   @return The maximum value in A
    */
    static float max(Tensor* A);

    /**
    *   @brief Obtain the maximum value in a tensor
    *   @param axis Vector with the axis in which to obtain the maximum value
    *   @param keepdims If true, output tensor will have the same dimentions as input tensor, except from the axis selected where dimension will be 1.
    *   @return float. The maximum value in A on the selected axis
    */
    Tensor* max(vector<int> axis, bool keepdims);


    static void max(Tensor* A, Tensor *B, ReduceDescriptor2 *rd);

    /**
    *   @brief Obtain the index of the maximum value in the tensor
    *   @return The desired index.
    */
    int argmax();

    /**
    *   @brief Obtain the index of the maximum value in the tensor
    *   @param A The tensor where the operation is applied.
    *   @return The desired index.
    */
    static int argmax(Tensor* A);

    /**
    *   @brief Obtain the index of the maximum value in a tensor
    *   @param axis Vector with the axis in which to obtain the maximum value
    *   @param keepdims If true, output tensor will have the same dimentions as input tensor, except from the axis selected where dimension will be 1.
    *   @return float. The tensor with the indexes of the maximum values in A on the selected axis
    */
    Tensor* argmax(vector<int> axis, bool keepdims);

    static void argmax(Tensor* A, Tensor *B, ReduceDescriptor2 *rd);

    /**
    *   @brief Obtain the minimum value in the tensor
    *   @return float. The minimum value in the tensor
    */
    float min();

    /**
    *   @brief Obtain the minimum value in a tensor
    *   @param A The tensor where the operation is applied
    *   @return The minimum value in A
    */
    static float min(Tensor* A);

    /**
    *   @brief Obtain the minimum value in a tensor
    *   @param axis Vector with the axis in which to obtain the minimum value
    *   @param keepdims If true, output tensor will have the same dimentions as input tensor, except from the axis selected where dimension will be 1.
    *   @return float. The minimum value in A on the selected axis
    */
    Tensor* min(vector<int> axis, bool keepdims);
    static void min(Tensor* A, Tensor *B, ReduceDescriptor2 *rd);

    /**
    *   @brief Obtain the index of the minimum value in the tensor
    *   @return The desired index.
    */
    int argmin();

    /**
    *   @brief Obtain the index of the minimum value in the tensor
    *   @param A The tensor where the operation is applied.
    *   @return The desired index.
    */
    static int argmin(Tensor* A);

    /**
    *   @brief Obtain the index of the minimum value in a tensor
    *   @param axis Vector with the axis in which to obtain the minimum value
    *   @param keepdims If true, output tensor will have the same dimentions as input tensor, except from the axis selected where dimension will be 1.
    *   @return float. The tensor with the indexes of the minimum values in A on the selected axis
    */
    Tensor* argmin(vector<int> axis, bool keepdims);
    static void argmin(Tensor* A, Tensor *B, ReduceDescriptor2 *rd);

    /**
    *   @brief Obtain the sum of all the values in the tensor.
    *   @return The sum of all the elements in the tensor.
    */
    float sum();

    /**
    *   @brief Obtain the sum of all the values in a tensor.
    *   @param A Input tensor.
    *   @return The sum of all the elements in the input tensor.
    */
    static float sum(Tensor* A);

    /**
    *   @brief Obtain the sum of all the element in the tensor
    *   @param axis Vector with the axis in which to obtain the sum value
    *   @param keepdims If true, output tensor will have the same dimentions as input tensor, except from the axis selected where dimension will be 1.
    *   @return float. The tensor with the sum of elements in A on the selected axis.
    */
    Tensor* sum(vector<int> axis, bool keepdims);
    static void sum(Tensor* A, Tensor *B, ReduceDescriptor2 *rd);

    /**
    *   @brief Obtain the absolute value sum of all the values in the tensor.
    *   @return The absolute value sum of all the elements in the tensor.
    */
    float sum_abs();

    /**
    *   @brief Obtain the absolute value sum of all the values in a tensor.
    *   @param A Input tensor.
    *   @return The absolute value sum of all the elements in the input tensor.
    */
    static float sum_abs(Tensor* A);

    /**
    *   @brief Obtain the absolute value sum of all the element in the tensor
    *   @param axis Vector with the axis in which to obtain the absolute value sum
    *   @param keepdims If true, output tensor will have the same dimentions as input tensor, except from the axis selected where dimension will be 1.
    *   @return float. The tensor with the absolute value sum of elements in A on the selected axis.
    */
    Tensor* sum_abs(vector<int> axis, bool keepdims);
    static void sum_abs(Tensor* A, Tensor *B, ReduceDescriptor2 *rd);

    /**
    *   @brief Obtain the product of all the values in the tensor.
    *   @return The product of all the elements in the tensor.
    */
    float prod();

    /**
    *   @brief Obtain the product of all the values in a tensor.
    *   @param A Input tensor.
    *   @return The product of all the elements in the input tensor.
    */
    static float prod(Tensor* A);

    /**
    *   @brief Obtain the product of all the element in the tensor
    *   @param axis Vector with the axis in which to obtain the product
    *   @param keepdims If true, output tensor will have the same dimentions as input tensor, except from the axis selected where dimension will be 1.
    *   @return float. The tensor with the product of elements in A on the selected axis.
    */
    Tensor* prod(vector<int> axis, bool keepdims);
    static void prod(Tensor* A, Tensor *B, ReduceDescriptor2 *rd);


    /**
    *   @brief Obtain the mean of all the values in the tensor.
    *   @return The mean of all the elements in the tensor.
    */
    float mean();

    /**
    *   @brief Obtain the mean of all the values in a tensor.
    *   @param A Input tensor.
    *   @return The mean of all the elements in the input tensor.
    */
    static float mean(Tensor* A);

    /**
    *   @brief Obtain the prodmeanuct of all the element in the tensor
    *   @param axis Vector with the axis in which to obtain the mean
    *   @param keepdims If true, output tensor will have the same dimentions as input tensor, except from the axis selected where dimension will be 1.
    *   @return float. The tensor with the mean of elements in A on the selected axis.
    */
    Tensor* mean(vector<int> axis, bool keepdims);
    static void mean(Tensor* A, Tensor *B, ReduceDescriptor2 *rd);


    /**
    *   @brief Obtain the standard deviation of all the values in the tensor.
    *   @param ubiased Whether the standard deviation is computed using the unbiased estimation or not.
    *   @return The standard deviation of all the elements in the tensor.
    */
    float std(bool unbiased=true);

    /**
    *   @brief Obtain the standard deviation of all the values in a tensor.
    *   @param A Input tensor.
    *   @param ubiased Whether the standard deviation is computed using the unbiased estimation or not.
    *   @return The standard deviation of all the elements in the input tensor.
    */
    static float std(Tensor* A, bool unbiased=true);

    /**
    *   @brief Obtain the standard deviation of all the elements in the tensor
    *   @param axis Vector with the axis in which to obtain the standard deviation
    *   @param keepdims If true, output tensor will have the same dimentions as input tensor, except from the axis selected where dimension will be 1.
    *   @param ubiased Whether the standard deviation is computed using the unbiased estimation or not.
    *   @return float. The tensor with the standard deviation of elements in A on the selected axis.
    */
    Tensor* std(vector<int> axis, bool keepdims, bool unbiased=true);
    static void std(Tensor* A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased=true);

    /**
    *   @brief Obtain the variance of all the values in the tensor.
    *   @param ubiased Whether the variance is computed using the unbiased estimation or not.
    *   @return The variance of all the elements in the tensor.
    */
    float var(bool unbiased=true);

    /**
    *   @brief Obtain the variance of all the values in a tensor.
    *   @param A Input tensor.
    *   @param ubiased Whether the variance is computed using the unbiased estimation or not.
    *   @return The variance of all the elements in the input tensor.
    */
    static float var(Tensor* A, bool unbiased=true);

    /**
    *   @brief Obtain the variance of all the elements in the tensor
    *   @param axis Vector with the axis in which to obtain the variance.
    *   @param keepdims If true, output tensor will have the same dimentions as input tensor, except from the axis selected where dimension will be 1.
    *   @param ubiased Whether the variance is computed using the unbiased estimation or not.
    *   @return float. The tensor with the variance of elements in A on the selected axis.
    */
    Tensor* var(vector<int> axis, bool keepdims, bool unbiased=true);
    static void var(Tensor* A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased=true);

    /**
    *   @brief Obtain the mode of all the values in the tensor.
    *   @return The mode of all the elements in the tensor.
    */
    int mode();

    /**
    *   @brief Obtain the mode of all the values in a tensor.
    *   @param A Input tensor.
    *   @return The mode of all the elements in the input tensor.
    */
    static int mode(Tensor* A);

    /**
    *   @brief Obtain the mode of all the elements in the tensor
    *   @param axis Vector with the axis in which to obtain the mode
    *   @param keepdims If true, output tensor will have the same dimentions as input tensor, except from the axis selected where dimension will be 1.
    *   @return float. The tensor with the mode of elements in A on the selected axis.
    */
    Tensor* mode(vector<int> axis, bool keepdims);
    static void mode(Tensor* A, Tensor *B, ReduceDescriptor2 *rd);

    // Math operations (unary) ************************
    /**
    *   @brief Inplace element-wise abs operation
    */
    void abs_();

    /**
    *   @brief Element-wise abs operation
    *   @param A The tensor where the operation is applied
    *   @return A new tensor with abs applied over A
    */
    Tensor* abs();

    /**
    *   @brief Element-wise abs operation
    *   @param A The tensor where the operation is applied
    *   @param B A new tensor with abs applied over A
    */
    static void abs(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise acos operation
    */
    void acos_();

    /**
    *   @brief Inplace element-wise acos operation
    *   @param A. The tensor where the operation is applied
    *   @return A new tensor with abs applied over A
    */
    Tensor* acos();

    /**
    *   @brief Element-wise acos operation
    *   @param A The tensor where the operation is applied
    *   @param B A new tensor with acos applied over A
    */
    static void acos(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise add operation of a tensor and a real value
    *   @param v. The real number to add
    */
    void add_(float v);

    /**
    *   @brief Element-wise add operation of a tensor and a real value
    *   @param v. The real number to add
    *   @return A new tensor with the sum
    */
    Tensor* add(float v);

    /**
    *   @brief Inplace element-wise add operation of two tensors
    *   @param A The tensor to be added.
    */
    void add_(Tensor* A);  // this = this .+ A

    /**
    *   @brief Element-wise add operation of two tensors
    *   @param A The tensor to be added
    *   @return a tensor with the element-wise sum
    */
    Tensor* add(Tensor* A);  // this = this .+ A

    /**
    *   @brief Element-wise add operation of a tensor and a real value
    *   @param A Input tensor
    *   @param B Output tensor. B = A + v
    *   @param v Real value to be added to A
    */
    static void add(Tensor *A, Tensor *B, float v); // B = A + v

    /**
    *   @brief Inplace element-wise asin operation
    */
    void asin_();

    /**
    *   @brief Element-wise asin operation
    *   @return A new tensor with the result
    */
    Tensor* asin();

    /**
    *   @brief Element-wise asin operation
    *   @param A The tensor where the operation is applied
    *   @param B A new tensor with asin applied over A
    */
    static void asin(Tensor *A, Tensor *B);


    /**
    *   @brief Inplace element-wise atan operation
    */
    void atan_();

    /**
    *   @brief Element-wise atan operation
    *   @return A new tensor with the result
    */
    Tensor* atan();

    /**
    *   @brief Element-wise atan operation
    *   @param A The tensor where the operation is applied
    *   @param B A new tensor with atan applied over A
    */
    static void atan(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise ceil operation
    */
    void ceil_();

    /**
    *   @brief Element-wise ceil operation
    *   @return A new tensor with the result
    */
    Tensor* ceil();

    /**
    *   @brief Element-wise ceil operation
    *   @param A The tensor where the operation is applied
    *   @param B A new tensor with ceil applied over A
    */
    static void ceil(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace clamp all elements in the input tensor to the range [min, max].
    *   @param min The lower bound of the clamping range.
    *   @param max The upper bound of the clamping range.
    */
    void clamp_(float min, float max);

    /**
    *   @brief Clamp all elements in the input tensor to the range [min, max].
    *   @param min The lower bound of the clamping range.
    *   @param max The upper bound of the clamping range.
    *   @return A new tensor with the clamped values in the input tensor.
    */
    Tensor* clamp(float min, float max);

    /**
    *   @brief Clamp all elements in the input tensor to the range [min, max].
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor with the result.
    *   @param min The lower bound of the clamping range.
    *   @param max The upper bound of the clamping range.
    */
    static void clamp(Tensor *A, Tensor *B, float min, float max);

    /**
    *   @brief Inplace clamp all elements in the input tensor to the range [-infty, max].
    *   @param max The upper bound of the clamping range.
    */
    void clampmax_(float max);

    /**
    *   @brief Clamp all elements in the input tensor to the range [-infty, max].
    *   @param max The upper bound of the clamping range.
    *   @return A new tensor with the clamped values.
    */
    Tensor* clampmax(float max);

    /**
    *   @brief Clamp all elements in the input tensor to the range [-infty, max].
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor.
    *   @param max The upper bound of the clamping range.
    */
    static void clampmax(Tensor *A, Tensor *B, float max);

    /**
    *   @brief Inplace clamp all elements in the input tensor to the range [min, +infty].
    *   @param min The lower bound of the clamping range.
    */
    void clampmin_(float min);

    /**
    *   @brief Clamp all elements in the input tensor to the range [min, +infty].
    *   @param min The lower bound of the clamping range.
    *   @return A new tensor with the clamped values.
    */
    Tensor* clampmin(float min);

    /**
    *   @brief Clamp all elements in the input tensor to the range [min, +infty].
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor.
    *   @param min The lower bound of the clamping range.
    */
    static void clampmin(Tensor *A, Tensor *B, float min);

    /**
    *   @brief Inplace element-wise cos operation
    */
    void cos_();

    /**
    *   @brief Element-wise cos operation
    *   @return A new tensor with cos applied.
    */
    Tensor* cos();

    /**
    *   @brief Element-wise cos operation
    *   @param A The tensor where the operation is applied
    *   @param B The output tensor.
    */
    static void cos(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise cosh operation
    */
    void cosh_();

    /**
    *   @brief Element-wise cosh operation
    *   @return A new tensor with cosh applied.
    */
    Tensor* cosh();

    /**
    *   @brief Element-wise cosh operation
    *   @param A The tensor where the operation is applied
    *   @param B The output tensor.
    */
    static void cosh(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise division operation of a tensor and a real value
    *   @param v The real number to divide by
    */
    void div_(float v);

    /**
    *   @brief Element-wise division operation of a tensor and a real value
    *   @param v The real number to divide by.
    *   @return A new tensor with the division.
    */
    Tensor* div(float v);

    /**
    *   @brief Inplace element-wise division operation of two tensors
    *   @param A. The tensor to divide by
    */
    void div_(Tensor* A); // this = this ./ A

    /**
    *   @brief Inplace element-wise division operation of two tensors
    *   @param A The tensor to divide by
    *   @return A new tensor with the division.
    */
    Tensor* div(Tensor* A); // this = this ./ A

    /**
    *   @brief Element-wise division operation of a tensor and a real value.
    *   @param A The tensor where the operation is applied
    *   @param B The output tensor. B = A / v
    *   @param v The real number to divide by.
    */
    static void div(Tensor *A, Tensor *B, float v); // B = A / v

    /**
    *   @brief Inplace element-wise exp operation of a tensor
    */
    void exp_();

    /**
    *   @brief Element-wise exp operation of a tensor
    *   @return A new tensor with the exp operation applied
    */
    Tensor* exp();

    /**
    *   @brief Element-wise exp operation of a tensor
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor.
    */
    static void exp(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise floor operation
    */
    void floor_();

    /**
    *   @brief Element-wise floor operation
    *   @return A new tensor with the floor operation applied.
    */
    Tensor* floor();

    /**
    *   @brief Element-wise floor operation.
    *   @param A. The tensor where the operation is applied.
    *   @param B The output tensor.
    */
    static void floor(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise 1/x operation
    *   @param v the value multiplying the inverse
    */
    void inv_(float v=1.0f);

    /**
    *   @brief Element-wise 1/x operation
    *   @param v the value multiplying the inverse
    *   @return A new tensor with the result.
    */
    Tensor* inv(float v=1.0f);

     /**
    *   @brief Element-wise 1/x operation
    *   @param A The input tensor.
    *   @param B The output tensor.
    *   @param v the value multiplying the inverse.
    */
    static void inv(Tensor *A, Tensor *B, float v=1.0f);

    /**
    *   @brief Inplace element-wise log operation
    */
    void log_();

    /**
    *   @brief Element-wise log operation
    *   @return A new tensor with the log operation applied
    */
    Tensor* log();

    /**
    *   @brief Element-wise log operation
    *   @param A The tensor where the operation is applied
    *   @param B The output tensor.
    */
    static void log(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise log2 operation
    */
    void log2_();

    /**
    *   @brief Element-wise log2 operation
    *   @return A new tensor with the log2 operation applied.
    */
    Tensor* log2();

    /**
    *   @brief Element-wise log2 operation.
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor.
    */
    static void log2(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise log10 operation
    */
    void log10_();

    /**
    *   @brief Element-wise log10 operation
    *   @return A new tensor with the log10 operation applied.
    */
    Tensor* log10();

    /**
    *   @brief Element-wise log10 operation.
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor.
    */
    static void log10(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise logn operation.
    *   @param n The base of the logarithm.
    */
    void logn_(float n);

    /**
    *   @brief Element-wise logn operation.
    *   @param n The base of the logarithm.
    *   @return A new tensor with the logn operation applied.
    */
    Tensor* logn(float n);

    /**
    *   @brief Element-wise logn operation.
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor.
    *   @param n The base of the logarithm.
    */
    static void logn(Tensor *A, Tensor *B, float n);

    /**
    *   @brief Inplace element-wise mod operation.
    *   @param v The mod operator
    */
    void mod_(float v);

    /**
    *   @brief Element-wise mod operation.
    *   @param v The mod operator
    *   @return A new tensor with the operation applied.
    */
    Tensor* mod(float v);

    /**
    *   @brief Element-wise mod operation.
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor.
    *   @param v The mod operator.
    */
    static void mod(Tensor *A, Tensor *B, float v);

    /**
    *   @brief Inplace multiplication operation of a tensor by a scalar.
    *   @param v. The value to multiply by
    */
    void mult_(float v);

    /**
    *   @brief Multiplication operation of a tensor by a scalar.
    *   @param v The value to multiply by
    *   @return A tensor with the result
    */
    Tensor* mult(float v);

    /**
    *   @brief Inplace element-wise  multiplication operation of two 1D tensors.
    *   @param A The tensor to multiply by.
    */
    void mult_(Tensor* A); // this = this .* A

    /**
    *   @brief Element-wise multiplication operation of two 1D tensors.
    *   @param A The tensor to multiply by.
    *   @return A tensor with the result.
    */
    Tensor* mult(Tensor* A); // this = this .* A

    /**
    *   @brief Element-wise multiplication operation of a tensor and a real value.
    *   @param A The input tensor.
    *   @param B The output tensor. B = A * v.
    *   @param v The value to multiply by.
    */
    static void mult(Tensor *A, Tensor *B, float v); // B = A * v

    /**
    *   @brief Inplace element-wise change of sign operation.
    */
    void neg_();

    /**
    *   @brief Element-wise change of sign operation.
    *   @return A tensor with the result.
    */
    Tensor* neg();

    /**
    *   @brief Element-wise change of sign operation.
    *   @param A The tensor where the operation is applied.
    *   @param B A tensor with -A
    */
    static void neg(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise normalization of values in a given range.
    *   @param min. The lower bound of the new range
    *   @param max. The upper bound of the new range
    */
    void normalize_(float min=0.0f, float max=1.0f);

    /**
    *   @brief Inplace element-wise normalization of values in a given range.
    *   @param min. The lower bound of the new range.
    *   @param max. The upper bound of the new range.
    *   @return A tensor with the result.
    */
    Tensor* normalize(float min=0.0f, float max=1.0f);

    /**
    *   @brief Inplace element-wise normalization of values in a given range.
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor.
    *   @param min. The lower bound of the new range
    *   @param max. The upper bound of the new range
    */
    static void normalize(Tensor *A, Tensor *B, float min=0.0f, float max=1.0f);


    /**
    *   @brief Inplace element-wise power operation with base e.
    *   @param exp The exponent
    */
    void pow_(float exp);

    /**
    *   @brief Element-wise power operation with base e.
    *   @param exp. The exponent.
    *   @return A tensor with the result.
    */
    Tensor* pow(float exp);

    /**
    *   @brief Element-wise power operation with base e.
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor.
    *   @param exp The exponent
    */
    static void pow(Tensor *A, Tensor *B, float exp);

    /**
    *   @brief Inplace element-wise power operation.
    *   @param base The base of the power
    */
    void powb_(float base);

    /**
    *   @brief Element-wise power operation.
    *   @param base The base of the power.
    *   @return A tensor with the result.
    */
    Tensor* powb(float base);

    /**
    *   @brief Element-wise power operation.
    *   @param A the input tensor.
    *   @param B The output tensor.
    *   @param base The base of the power.
    */
    static void powb(Tensor *A, Tensor *B, float base);

    /**
    *   @brief Inplace element-wise reciprocal operation.
    */
    void reciprocal_();

    /**
    *   @brief Element-wise reciprocal operation.
    *   @return A tensor with the result
    */
    Tensor* reciprocal();

    /**
    *   @brief Element-wise reciprocal operation.
    *   @param A. The tensor where the operation is applied.
    *   @param B A tensor with reciprocal(A), element-wise
    */
    static void reciprocal(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise reminder operation.
    *   @param v. The real to divide A by
    */
    void remainder_(float v);

    /**
    *   @brief Element-wise reminder operation.
    *   @param v The real to divide A by
    *   @return A tensor with A%v
    */
    Tensor* remainder(float v);  // TODO: difference with mod??

    /**
    *   @brief Element-wise reminder operation.
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor.
    *   @param v The real to divide A by.
    */
    static void remainder(Tensor *A, Tensor *B, float v);

    /**
    *   @brief Inplace element-wise round operation.
    */
    void round_();

    /**
    *   @brief Element-wise round operation.
    *   @return A tensor with A rounded
    */
    Tensor* round();

    /**
    *   @brief Element-wise round operation.
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor.
    */
    static void round(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise inverse square root operation.
    */
    void rsqrt_();

    /**
    *   @brief Element-wise inverse square root operation.
    *   @return A tensor with the result
    */
    Tensor* rsqrt();

    /**
    *   @brief Element-wise inverse square root operation.
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor.
    */
    static void rsqrt(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise sigmoid operation.
    */
    void sigmoid_();

    /**
    *   @brief Element-wise sigmoid operation.
    *   @return A tensor with sigmoid(A)
    */
    Tensor* sigmoid();

    /**
    *   @brief Element-wise sigmoid operation.
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor.
    */
    static void sigmoid(Tensor *A, Tensor *B);

    void sign_(float zero_sign=0.0f);
    Tensor* sign(float zero_sign=0.0f);
    static void sign(Tensor *A, Tensor *B, float zero_sign=0.0f);

    /**
    *   @brief Inplace element-wise sin operation.
    */
    void sin_();

    /**
    *   @brief Element-wise sin operation.
    *   @return A tensor with the result.
    */
    Tensor* sin();

    /**
    *   @brief Element-wise sin operation.
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor.
    */
    static void sin(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise sinh operation.
    */
    void sinh_();

    /**
    *   @brief Element-wise sinh operation.
    *   @return A tensor with the result.
    */
    Tensor* sinh();

    /**
    *   @brief Element-wise sinh operation.
    *   @param A The tensor where the operation is applied.
    *   @param B Tensor with the result.
    */
    static void sinh(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise square operation. More efficient than using pow_(A, 2).
    */
    void sqr_();

    /**
    *   @brief Element-wise square operation. More efficient than using pow(A, 2).
    *   @return A tensor with the result
    */
    Tensor* sqr();

    /**
    *   @brief Element-wise square operation. More efficient than using pow(A, 2).
    *   @param A The tensor where the operation is applied.
    *   @param B tensor with the result.
    */
    static void sqr(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise square root operation.
    */
    void sqrt_();

    /**
    *   @brief Element-wise square operation.
    *   @return A tensor with the result.
    */
    Tensor* sqrt();

    /**
    *   @brief Element-wise square operation.
    *   @param A The tensor where the operation is applied.
    *   @param B tensor with the result.
    */
    static void sqrt(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise substraction operation of a tensor and a scalar.
    *   @param v The value to substract to A.
    */
    void sub_(float v);


    /**
    *   @brief Element-wise substraction operation of a tensor and a scalar.
    *   @param v The value to substract to the input tensor.
    *   @return A tensor with the result.
    */
    Tensor* sub(float v);

    /**
    *   @brief Inplace element-wise substraction operation of two tensors.
    *   @param A. The tensor to substract.
    */
    void sub_(Tensor* A); // this = this .- A

    /**
    *   @brief Element-wise substraction operation of two tensors.
    *   @param A The tensor to substract.
    *   @return A tensor with the result.
    */
    Tensor* sub(Tensor* A); // this = this .- A

    /**
    *   @brief Element-wise substraction operation of a tensor and a real value.
    *   @param A The tensor where the operation is applied.
    *   @param B The output tensor. B = A - v.
    *   @param v The real value to substract.
    */
    static void sub(Tensor *A, Tensor *B, float v);

    /**
    *   @brief Inplace element-wise tan operation.
    */
    void tan_();

    /**
    *   @brief Element-wise tan operation.
    *   @return A tensor with the result.
    */
    Tensor* tan();

    /**
    *   @brief Element-wise tan operation.
    *   @param A The tensor where the operation is applied.
    *   @param B A tensor with the result.
    */
    static void tan(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise tanh operation.
    */
    void tanh_();

    /**
    *   @brief Element-wise tanh operation.
    *   @return A tensor with the result.
    */
    Tensor* tanh();

    /**
    *   @brief Element-wise tanh operation.
    *   @param A The tensor where the operation is applied.
    *   @param B A tensor with the result.
    */
    static void tanh(Tensor *A, Tensor *B);

    /**
    *   @brief Inplace element-wise truncate operation.
    */
    void trunc_();

    /**
    *   @brief Element-wise truncate operation.
    *   @return A tensor with the result.
    */
    Tensor* trunc();

    /**
    *   @brief Element-wise truncate operation.
    *   @param A The tensor where the operation is applied.
    *   @param B tensor with the result.
    */
    static void trunc(Tensor *A, Tensor *B);


    // Math operations (binary) ************************
    static Tensor* add(Tensor *A, Tensor *B); // (new)C = A + B
    static void add(Tensor *A, Tensor *B, Tensor *C); // C = A + B

    static Tensor* div(Tensor *A, Tensor *B); // (new)C = A / B
    static void div(Tensor *A, Tensor *B, Tensor *C); // C = A / B

    static Tensor* mult(Tensor *A, Tensor *B); // (new)C = A * B
    static void mult(Tensor *A, Tensor *B, Tensor *C); // C = A * B

    static Tensor* interpolate(float factor1, Tensor *A, float factor2, Tensor *B); // (new)C = f1*A + f2*B
    static void interpolate(float factor1, Tensor *A, float factor2, Tensor *B, Tensor *C);  // C = f1*A + f2*B

    static Tensor* sub(Tensor *A, Tensor *B); // (new)C = A - B
    static void sub(Tensor *A, Tensor *B, Tensor *C); // C = A - B



    // ***** Core *****************************
    void fill_(float v);
    static void fill(Tensor* A, float v);

    void permute_(const vector<int>& dims);
    static Tensor* permute(Tensor* A, const vector<int>& dims);

    void moveaxis_(int source, int destination);
    static Tensor* moveaxis(Tensor* A, int source, int destination);

    void swapaxis_(int axis1, int axis2);
    static Tensor* swapaxis(Tensor* A, int axis1, int axis2);

    void reshape_(const vector<int> &new_shape);
    static Tensor* reshape(Tensor *A, const vector<int> &shape);

    void flatten_();
    static Tensor* flatten(Tensor *A);

    void squeeze_();
    static Tensor* squeeze(Tensor *A);

    void unsqueeze_();
    static Tensor* unsqueeze(Tensor *A);


    // ***** Transformations *****************************
    static void shift(Tensor *A,Tensor *B, vector<int> shift, WrappingMode mode=WrappingMode::Constant, float cval=0.0f);
    static void rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center={0,0}, WrappingMode mode=WrappingMode::Constant, float cval=0.0f);
    static void scale(Tensor *A, Tensor *B, vector<int> new_shape, WrappingMode mode=WrappingMode::Nearest, float cval=0.0f);
    static void flip(Tensor *A, Tensor *B, int axis=0);
    static void crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float cval=0.0f);
    static void crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, WrappingMode mode=WrappingMode::Nearest, float cval=0.0f);
    static void cutout(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float cval=0.0f);

    // ***** Data augmentation *****************************
    static void shift_random(Tensor *A,Tensor *B, vector<float> factor_x, vector<float> factor_y, WrappingMode mode=WrappingMode::Constant, float cval=0.0f);
    static void rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center={0,0}, WrappingMode mode=WrappingMode::Constant, float cval=0.0f);
    static void scale_random(Tensor *A, Tensor *B, vector<float> factor, WrappingMode mode=WrappingMode::Nearest, float cval=0.0f);
    static void flip_random(Tensor *A, Tensor *B, int axis);

    static void crop_random(Tensor *A, Tensor *B);
    static void crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, WrappingMode mode=WrappingMode::Nearest, float cval=0.0f);
    static void cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float cval=0.0f);


    // Linear algebra *****************************
    float trace(int k=0);
    static float trace(Tensor *A, int k=0);

    float norm(string ord="fro");
    static float norm(Tensor *A, string ord="fro");
    Tensor* norm(vector<int> axis, bool keepdims, string ord="fro");
    static void norm(Tensor* A, Tensor *B, ReduceDescriptor2 *rd, string ord="fro");

    // Generating index arrays *****************************
    std::pair<unsigned int*, int> _nonzero();
    Tensor* nonzero(bool sort_indices=false);

    static Tensor* where(Tensor *condition, Tensor *A, Tensor *B);  // where(x > 0, x[random], y[ones])
    static void where(Tensor *condition, Tensor *A, Tensor *B, Tensor *C);

    Tensor* mask_indices(Tensor *mask, Tensor *A);  // where(x > 0, x[random], y[ones])
    static void mask_indices(Tensor *mask, Tensor *A, Tensor *B);


    // Logic funcions: Truth value testing *****************************

    /**
      *  @brief Test whether all elements evaluate to True.
      *
      *  @param A   Tensor to evaluate
      *  @return    bool
    */
    static bool all(Tensor *A);

    /**
      *  @brief Test whether any element evaluates to True.
      *
      *  @param A   Tensor to evaluate
      *  @return    bool
    */
    static bool any(Tensor *A);

    // Logic funcions: Logical ops

    /**
      *  @brief Test element-wise for finiteness (not infinity or not Not a Number).
      *
      *  @param A   Tensor to evaluate
      *  @param B   Tensor to store the results of the test as booleans
      *  @return    void
    */
    static void isfinite(Tensor *A, Tensor* B);

    /**
      *  @brief Test element-wise for positive or negative infinity.
      *
      *  @param A   Tensor to evaluate
      *  @param B   Tensor to store the results of the test as booleans
      *  @return    void
    */
    static void isinf(Tensor *A, Tensor* B);

    /**
      *  @brief Test element-wise for Nan.
      *
      *  @param A   Tensor to evaluate
      *  @param B   Tensor to store the results of the test as booleans
      *  @return    void
    */
    static void isnan(Tensor *A, Tensor* B);

    /**
      *  @brief Test element-wise for negative infinity.
      *
      *  @param A   Tensor to evaluate
      *  @param B   Tensor to store the results of the test as booleans
      *  @return    void
    */
    static void isneginf(Tensor *A, Tensor* B);

    /**
      *  @brief Test element-wise for positive infinity.
      *
      *  @param A   Tensor to evaluate
      *  @param B   Tensor to store the results of the test as booleans
      *  @return    void
    */
    static void isposinf(Tensor *A, Tensor* B);

    // Logic funcions: Logical ops

    /**
      *  @brief Compute the truth value of ``A and B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor to store the results of the operation
      *  @return    void
    */
    static void logical_and(Tensor *A, Tensor *B, Tensor *C);

    /**
      *  @brief Compute the truth value of ``A or B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor to store the results of the operation
      *  @return    void
    */
    static void logical_or(Tensor *A, Tensor *B, Tensor *C);

    /**
      *  @brief Compute the truth value of ``not A`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor to store the results of the operation
      *  @return    void
    */
    static void logical_not(Tensor *A, Tensor *B);

    /**
      *  @brief Compute the truth value of ``A xor B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor to store the results of the operation
      *  @return    void
    */
    static void logical_xor(Tensor *A, Tensor *B, Tensor *C);

    // Logic funcions: Comparison ops *****************************

    /**
      *  @brief Returns True if two arrays are element-wise equal within a tolerance.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param rtol
      *  @param atol
      *  @param equal_nan
      *  @return    void
    */
    static bool allclose(Tensor *A, Tensor *B, float rtol=1e-05, float atol=1e-08, bool equal_nan=false);  // Returns true or false

    /**
      *  @brief Returns a boolean array where two arrays are element-wise equal within a tolerance.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor
      *  @param rtol
      *  @param atol
      *  @param equal_nan
      *  @return    void
    */
    static void isclose(Tensor *A, Tensor *B, Tensor *C, float rtol=1e-05, float atol=1e-08, bool equal_nan=false);  // Returns a boolean tensor


    void greater_(float v);
    Tensor* greater(float v);
    static void greater(Tensor *A, Tensor *B, float v);

    Tensor* greater(Tensor *A);

    /**
      *  @brief Return the truth value of ``A > B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor store the results of the operation.
      *  @return    void
    */
    static void greater(Tensor *A, Tensor *B, Tensor *C);


    void greater_equal_(float v);
    Tensor* greater_equal(float v);
    static void greater_equal(Tensor *A, Tensor *B, float v);

    Tensor* greater_equal(Tensor *A);

    /**
      *  @brief Return the truth value of ``A >= B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor store the results of the operation.
      *  @return    void
    */
    static void greater_equal(Tensor *A, Tensor *B, Tensor *C);

    void less_(float v);
    Tensor* less(float v);
    static void less(Tensor *A, Tensor *B, float v);

    Tensor* less(Tensor *A);

    /**
      *  @brief Return the truth value of ``A < B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor store the results of the operation.
      *  @return    void
    */
    static void less(Tensor *A, Tensor *B, Tensor *C);

    void less_equal_(float v);
    Tensor* less_equal(float v);
    static void less_equal(Tensor *A, Tensor *B, float v);

    Tensor* less_equal(Tensor *A);

    /**
      *  @brief Return the truth value of ``A <= B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor store the results of the operation.
      *  @return    void
    */
    static void less_equal(Tensor *A, Tensor *B, Tensor *C);

    void equal_(float v);
    Tensor* equal(float v);
    static void equal(Tensor *A, Tensor *B, float v);

    Tensor* equal(Tensor *A);

    /**
      *  @brief Return the truth value of ``A == B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor store the results of the operation.
      *  @return    void
    */
    static void equal(Tensor *A, Tensor *B, Tensor *C);

    void not_equal_(float v);
    Tensor* not_equal(float v);
    static void not_equal(Tensor *A, Tensor *B, float v);

    Tensor* not_equal(Tensor *A);

    /**
      *  @brief Return the truth value of ``A != B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor store the results of the operation.
      *  @return    void
    */
    static void not_equal(Tensor *A, Tensor *B, Tensor *C);

    // Math operations: Other ops
    // TODO: cross, einsum, flip, dot, etc

    void sort_(bool descending=false, bool stable=true);
    Tensor* sort(bool descending=false, bool stable=true);
    static void sort(Tensor* A, Tensor* B, bool descending=false, bool stable=true);

    Tensor* argsort(bool descending=false, bool stable=true);
    static void argsort(Tensor* A, Tensor* B, bool descending=false, bool stable=true);

    // Indexing, Slicing, Joining, Mutating Ops *************
    static Tensor* concat(vector<Tensor*> A, unsigned int axis=0, Tensor* output=nullptr);
    static void concat_back(Tensor *A, vector<Tensor*> t, unsigned int axis);


    /**
      *  @brief Returns an array with the selected indices of the tensor.
      *
      *  @param indices  Vector of strings representing the indices to be selected. These indices must follow a Python-like syntax. Some examples: ``"0"`` , ``":5"`` , ``":"`` , ``"3:6"``.
      *  @return     Tensor
    */
    Tensor* select(const vector<string>& indices);
    static void select(Tensor *A, Tensor *B, SelDescriptor *sd);
    static void select_back(Tensor *A, Tensor *B, SelDescriptor *sd);

    /**
      *  @brief Sets the elements in the array using the selected indices. The indices must be specified as a vector of strings ({“0”, “:5”, “:”, “3:6”}).
      *
      *  @param indices  Vector of strings representing the indices to be selected. These indices must follow a Python-like syntax. Some examples: ``"0"``, ``":5"``, ``":"``, ``"3:6"``.
      *  @param A
      *  @return     void
    */
    void set_select(const vector<string>& indices, Tensor *A);
    static void set_select(Tensor *A, Tensor *B, SelDescriptor *sd);
    static void set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd);

    /**
     *  @brief Clone a tensor (same device). Similar to copy, but returning a new instance
     *
     *  @return    Tensor
    */
    Tensor* clone();

    /**
      *  @brief Reallocates a tensor into this one. Deprecated.
      *
      *  @return
    */
    void reallocate(Tensor* old_t, vector<int> *s = nullptr);

    /**
      *  @brief Resizes a tensor ({2, 2, 2} => {10, 2, 2}).
      *
      *  @return
    */
    void resize(int b, float *fptr=nullptr);


    // ***********************************************************
    // ***********************************************************
    // ***********************************************************
    // ************************ LEGACY ***************************
    // ***********************************************************
    // ***********************************************************
    // ***********************************************************

    // Generators (In-place) *************************************
    // TODO: Rethink names + static
    void rand_bernoulli(); // Todo
    void rand_multinomial(); // Todo
    void rand_uniform(float v);
    void rand_signed_uniform(float v);
    void rand_normal(float m, float s, bool fast_math=true);
    void rand_binary(float v);

    // ***** Overload operators *****************************
    // Tensor and Tensor (Element wise)
    friend Tensor& operator+ (Tensor &A, Tensor &B);
    friend Tensor& operator- (Tensor &A, Tensor &B);
    friend Tensor& operator* (Tensor &A, Tensor &B);
    friend Tensor& operator/ (Tensor &A, Tensor &B);

    // Tensor op= Tensor
    friend void operator+= (Tensor &A, Tensor &B);
    friend void operator-= (Tensor &A, Tensor &B);
    friend void operator*= (Tensor &A, Tensor &B);
    friend void operator/= (Tensor &A, Tensor &B);

    // Tensor op= scalar
    friend void operator+= (Tensor &A, float v);
    friend void operator-= (Tensor &A, float v);
    friend void operator*= (Tensor &A, float v);
    friend void operator/= (Tensor &A, float v);

    // Tensor and scalar
    friend Tensor& operator+ (Tensor &A, float v);
    friend Tensor& operator- (Tensor &A, float v);
    friend Tensor& operator* (Tensor &A, float v);
    friend Tensor& operator/ (Tensor &A, float v);

    // scalar and Tensor
    friend Tensor& operator+ (float v, Tensor &A);
    friend Tensor& operator- (float v, Tensor &A);
    friend Tensor& operator* (float v, Tensor &A);
    friend Tensor& operator/ (float v, Tensor &A);


    // *******************************************
    // Legacy ************************************
    // *******************************************


    /**
      *  @brief Copy data from tensor A to B.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @return    void
    */
    static void copy(Tensor *A, Tensor *B);
    static void fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc);
    static void select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end, bool mask_zeros=false);
    static void deselect(Tensor *A, Tensor *B, vector<int> sind, int ini, int end,int inc=0, bool mask_zeros=false);
    static void tile(Tensor *A, Tensor *B);

    // TODO: REFACTOR!!! ************************
    static void transpose(Tensor *A, Tensor *B, vector<int> dims);  // TODO: Should be replaced by permute
    static void add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC); // C = a*A+b*B
    static void inc(Tensor *A, Tensor *B);
    static void el_div(Tensor *A, Tensor *B, Tensor *C, int incC);
    static void el_mult(Tensor *A, Tensor *B, Tensor *C, int incC);
    static void mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC);
    static void sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
    static void sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);
    static void reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB);
    static int eqsize(Tensor *A, Tensor *B);  // Legacy
    static int sameShape(Tensor *A, Tensor *B);  // Previously named "Tensor::eqsize"
    static int equivalent(Tensor *A, Tensor *B, float epsilon=1e-3);  // Previously named "Tensor::equal2"

};



//template<typename T>
//Tensor* Tensor::load_from_numpy(const string &filename, const string &format){
//    Tensor* t = nullptr;
//
//    cnpy::NpyArray arr = cnpy::npy_load(filename);
//    auto* loaded_data = arr.data<T>();
//
//    // Get shape
//    vector<int> arr_shape;
//    for(unsigned long i : arr.shape){
//        arr_shape.push_back(i);
//    }
//
//    // Initialize tensor
//    t = new Tensor(arr_shape, DEV_CPU);
//
//    // Fill tensor
//    for(int i=0; i<arr.num_vals; i++){
//        t->ptr[i] = (float)loaded_data[i];
//    }
//
//    return t;
//}


template<typename T>
Tensor* Tensor::load(const string& filename, string format){
    // Infer format from filename
    if(format.empty()){
        format = get_extension(filename);
    }

    // Check if file exists (open file stream)
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.good()){
        msg("File not found. Check the file name and try again.", "Tensor::load");
    }
    // Load tensor
    Tensor* t;
    if(format=="jpg" || format=="jpeg" || format=="png" || format=="bmp" ||
       format=="hdr" || format=="psd" || format=="tga" || format=="gif" ||
       format=="pic"  || format=="pgm"  || format=="ppm") { // Images
        t = Tensor::load_from_img(filename, format);
    }else if(format=="bin" || format=="onnx"){
        t = Tensor::loadfs(ifs, format);
    }else if(format=="npy" || format=="npz"){  // Deprecated
        msg("Format deprecated in favor of python: *.'" + format + "'", "Tensor::load");
//        t = Tensor::load_from_numpy<T>(filename, format);
    }else if(format=="csv" || format=="tsv" || format=="txt"){  // Deprecated
        msg("Format deprecated in favor of python: *.'" + format + "'", "Tensor::load");
//        t = Tensor::loadfs(ifs, format);
    }else{
        msg("Format not implemented: *.'" + format + "'", "Tensor::load");
    }

    // Close file stream and return tensor
    ifs.close();
    return t;
}

void checkCompatibility(Tensor *A, Tensor *B, const string &title);
void checkCompatibility(Tensor *A, Tensor *B, Tensor *C, const string &title);

#endif //EDDL_TENSOR_H
