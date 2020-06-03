///*
//* EDDL Library - European Distributed Deep Learning Library.
//* Version: 0.6
//* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
//* Date: April 2020
//* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
//* All rights reserved
//*/
//
//
//#ifndef EDDL_T_H
//#define EDDL_T_H
//
//
//#include "eddl/tensor/tensor.h"
//#include "eddl/tensor/tensor_reduction.h"
//
//typedef Tensor* tensor;
//typedef vector<int> tshape;
//
//namespace eddlT{
//
//    // Creation ops ***********************************
//
//    /**
//    *  @brief Constructor of an uninitialized tensor
//    *
//    *  @param shape Vector of ints specifying the shape of the tensor
//    *  @return a tensor
//    */
//    Tensor* create(const vector<int> &shape);
//
//    /**
//    *  @brief Constructor of an uninitialized tensor
//    *
//    *  @param shape Vector of ints specifying the shape of the tensor
//    *  @param dev  One of DEV_CPU or DEV_GPU
//    *  @return a tensor
//    */
//    Tensor* create(const vector<int> &shape, int dev);
//
//
//    Tensor* create(const vector<int> &shape, float *ptr);
//
//    /**
//    *  @brief Constructor of an uninitialized tensor
//    *
//    *  @param shape Vector of ints specifying the shape of the tensor
//    *  @param ptr  memory pointer
//    *  @param dev  One of DEV_CPU or DEV_GPU
//    *  @return a tensor
//    */
//    Tensor* create(const vector<int> &shape, float *ptr, int dev);
//
//
//    /**
//    *   @brief Create a tensor of the specified shape and fill it with zeros.
//    *
//    *   @param shape Shape of the tensor to create.
//    *   @param dev  Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
//    *   @return The zero-initialized tensor
//    */
//    Tensor* zeros(const vector<int> &shape, int dev=DEV_CPU);
//
//    /**
//    *   @brief Create a tensor of the specified shape and fill it with ones.
//    *
//    *   @param shape Shape of the tensor to create.
//    *   @param dev  Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
//    *   @return The ones-initialized tensor
//    */
//    Tensor* ones(const vector<int> &shape, int dev=DEV_CPU);
//
//    /**
//    *   @brief Create a tensor of the specified shape and fill it with a value.
//    *
//    *   @param shape Shape of the tensor to create.
//    *   @param value  Value to fill the tensor with.
//    *   @param dev  Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
//    *   @return The initialized tensor
//    */
//    Tensor* full(const vector<int> &shape, float value, int dev=DEV_CPU);
//
//    /**
//    *   @brief Create a 1-D tensor of size ceil(end - start) with values from start to end with step step.
//    *
//    *   @param start Start index
//    *   @param end  End index
//    *   @param step  The gap between two values in the tensor.
//    *   @param dev One of ``DEV_CPU``or ``DEV_GPU``
//    *   @return The new tensor
//    */
//    Tensor* arange(float start, float end, float step, int dev=DEV_CPU);
//
//    /**
//    *   @brief Creates a 1-D tensor of size floor(end - start)/ + 1 with values from start to end with step step.
//    *
//    *   @param start Start value
//    *   @param end  End value
//    *   @param step  The gap between two values in the tensor.
//    *   @param dev One of ``DEV_CPU``or ``DEV_GPU``
//    *   @return The new tensor
//    */
//    Tensor* range(float start, float end, float step, int dev=DEV_CPU);
//
//    /**
//    *   @brief Creates a 1-D tensor with a sequence of num evenly-spaced values starting at start. If steps > 1, the values in the sequence increase by end - start / steps - 1, so that the last one is exactly end.
//    *   @param start Start value
//    *   @param end  End value
//    *   @param steps  The gap between two values in the tensor.
//    *   @param dev One of ``DEV_CPU``or ``DEV_GPU``
//    *   @return The new tensor
//    */
//    Tensor* linspace(float start, float end, int steps, int dev=DEV_CPU);
//
//
//    Tensor* logspace(float start, float end, int steps, float base, int dev=DEV_CPU);
//
//
//    Tensor* eye(int size, int dev=DEV_CPU);
//    Tensor* randn(const vector<int> &shape, int dev=DEV_CPU);
//
//
//    // Copy data     c   ********************************
//    /**
//    *   @brief Move a tensor to CPU
//    *   @param A The tensor to move
//    */
//    void toCPU_(Tensor *A);
//
//    /**
//    *   @brief Move a tensor to GPY
//    *   @param A The tensor to move
//    */
//    void toGPU_(Tensor *A);
//
//    /**
//    *   @brief Move a copy of a tensor to CPU
//    *   @param A The tensor to clone and move
//    *   @return The cloned tensor in CPU
//    */
//    Tensor * toCPU(Tensor *A);
//
//    /**
//    *   @brief Move a copy of a tensor to GPU
//    *   @param A The tensor to clone and move
//    *   @return The cloned tensor in GPU
//    */
//    Tensor * toGPU(Tensor *A);
//
//    /**
//    *   @brief Obtain the copy of a tensor
//    *   @param A The tensor to clone
//    *   @return The cloned tensor
//    */
//    Tensor* clone(Tensor *A);
//
//
//    Tensor* select(Tensor *A, int i);
//
//    /**
//    *   @brief Copy tensor A in tensor B
//    *   @param A The tensor to copy
//    *   @param B The destination tensor
//    */
//    void copyTensor(Tensor *A,Tensor *B);
//
//    // Core inplace    **********************************
//    /**
//    *   @brief Fill tensor with a value
//    *   @param A The tensor to fill
//    *   @param v the value to fill the tensor with
//    */
//    void fill_(Tensor *A,float v);
//
//    /**
//    *   @brief Substitute some positions of a tensor with a value
//    *   @param A. The tensor to modify
//    *   @param indices. The vector of indices to be modified
//    *   @param value. The new value of the altered positions
//    */
//    void set_(Tensor *A,vector<int> indices, float value);
//
//    /**
//    *   @brief Reshape a tensor
//    *   @param A. The tensor to reshape
//    *   @param indices. The vector with the new tensor shape
//    */
//    void reshape_(Tensor *A, vector<int> indices);
//
//    // Pointer functions ********************************
//    /**
//    *   @brief Get pointet to a tensor
//    *   @param A. The tensor we want to obtain the pointer to
//    *   @return float. The pointer to tensor A
//    */
//    float *getptr(Tensor *A);
//
//    // Other functions   ********************************
//    void print(Tensor *A);
//    void info(Tensor *A);
//    tshape getShape(Tensor *A);
//
//    // Serialization ***********************************
//    /**
//    *   @brief Load content from file to a tensor
//    *   @param fname. The file path.
//    *   @param format. The format of the file. Supported image formats: jpg, jpeg, png, bmp, hdr, psd, tga, gif, pic, pgm, ppm, bin, onnx, npy, npz, csv, tsv, txt
//    *   @return The initialized tensor
//    */
//    Tensor* load(string fname, string format="");
//
//    template<typename T>
//    Tensor* load_from_numpy(const string &filename, const string &format=""){
//        msg("Format deprecated in favor of python: *.'" + format + "'", "Tensor::load_from_numpy");
//        return nullptr;
////        return Tensor::load_from_numpy<T>(filename, format);
//    }
//    template<typename T>
//    Tensor* load(const string& filename, string format=""){
//        return Tensor::load<T>(filename, format);
//    }
//
//    /**
//    *   @brief Save content from tensor to a file
//    *   @param A. The tensor to save
//    *   @param fname. The file path
//    *   @param format. The format of the file. Supported image formats: jpg, jpeg, png, bmp, hdr, psd, tga, gif, pic, pgm, ppm, bin, onnx, csv, tsv, txt
//    *   @return The initialized tensor
//    */
//    void save(Tensor* A, string fname, string format="");
//
//    // Math ops       ***********************************
//    /**
//    *   @brief Inplace element-wise abs operation
//    *   @param A. The tensor where the operation is applied
//    */
//    void abs_(Tensor *A);
//
//    /**
//    *   @brief Element-wise abs operation
//    *   @param A. The tensor where the operation is applied
//    *   @return A new tensor with abs applied over A
//    */
//    Tensor* abs(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise acos operation
//    *   @param A. The tensor where the operation is applied
//    */
//    void acos_(Tensor *A);
//
//     /**
//    *   @brief Element-wise acos operation
//    *   @param A. The tensor where the operation is applied
//    *   @return A new tensor with acos applied over A
//    */
//    Tensor* acos(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise add operation of a tensor and a real value
//    *   @param A. The tensor where the operation is applied
//    *   @param v. The real number to add
//    */
//    void add_(Tensor *A,float v);
//
//    /**
//    *   @brief Element-wise add operation of a tensor and a real value
//    *   @param A. The tensor where the operation is applied
//    *   @param v. The real number to add
//    *   @return A new tensor with the sum
//    */
//    Tensor *add(Tensor *A,float v);
//
//    /**
//    *   @brief Inplace element-wise add operation of two tensors
//    *   @param A. The tensor where the operation is applied
//    *   @param B. The tensor to add to A
//    */
//    void add_(Tensor *A,Tensor *B);
//
//    /**
//    *   @brief Element-wise add operation of two tensors
//    *   @param A. A tensor
//    *   @param B. Another tensor
//    *   @return a tensor with the element-wise sum A+B
//    */
//    Tensor* add(Tensor *A, Tensor *B);
//
//
//    /**
//    *   @brief Inplace element-wise asin operation
//    *   @param A. The tensor where the operation is applied
//    */
//    void asin_(Tensor *A);
//
//    /**
//    *   @brief Element-wise asin operation
//    *   @param A. The tensor where the operation is applied
//    *   @return A new tensor with asin applied over A
//    */
//    Tensor* asin(Tensor *A);
//
//
//
//    /**
//    *   @brief Inplace element-wise atan operation
//    *   @param A. The tensor where the operation is applied
//    */
//    void atan_(Tensor *A);
//
//    /**
//    *   @brief Element-wise atan operation
//    *   @param A. The tensor where the operation is applied
//    *   @return A new tensor with atan applied over A
//    */
//    Tensor* atan(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise ceil operation
//    *   @param A. The tensor where the operation is applied
//    */
//    void ceil_(Tensor *A);
//
//    /**
//    *   @brief Element-wise ceil operation
//    *   @param A. The tensor where the operation is applied
//    *   @return A new tensor with ceil applied over A
//    */
//    Tensor* ceil(Tensor *A);
//
//    void clamp_(Tensor *A,float min, float max);
//    Tensor* clamp(Tensor *A, float min, float max);
//
//    void clampmax_(Tensor *A,float max);
//    Tensor* clampmax(Tensor *A, float max);
//
//    void clampmin_(Tensor *A,float min);
//    Tensor* clampmin(Tensor *A, float min);
//
//    /**
//    *   @brief Inplace element-wise cos operation
//    *   @param A. The tensor where the operation is applied
//    */
//    void cos_(Tensor *A);
//
//    /**
//    *   @brief Element-wise cos operation
//    *   @param A. The tensor where the operation is applied
//    *   @return A new tensor with cos applied over A
//    */
//    Tensor* cos(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise cosh operation
//    *   @param A. The tensor where the operation is applied
//    */
//    void cosh_(Tensor *A);
//
//    /**
//    *   @brief Element-wise cosh operation
//    *   @param A. The tensor where the operation is applied
//    *   @return A new tensor with cosh applied over A
//    */
//    Tensor* cosh(Tensor *A);
//
//    void inv_(Tensor *A);
//
//
//    /**
//    *   @brief Inplace element-wise division operation of a tensor and a real value
//    *   @param A. The tensor where the operation is applied
//    *   @param v. The real number to divide by
//    */
//    void div_(Tensor *A,float v);
//
//    /**
//    *   @brief Element-wise division operation of a tensor and a real value
//    *   @param A. The tensor where the operation is applied
//    *   @param v. The real number to divide by
//    *   @return A new tensor with the division A/v
//    */
//    Tensor* div(Tensor *A, float v);
//
//    /**
//    *   @brief Inplace element-wise division operation of two tensors
//    *   @param A. The tensor where the operation is applied
//    *   @param B. The tensor to divide by
//    */
//    void div_(Tensor *A, Tensor *B);
//
//    /**
//    *   @brief Inplace element-wise division operation of two tensors
//    *   @param A. The tensor where the operation is applied
//    *   @param B. The tensor to divide by
//    *   @return A new tnesor with the division A/B element-wise
//    */
//    Tensor *div(Tensor *A, Tensor *B);
//
//    /**
//    *   @brief Inplace element-wise exp operation of a tensor
//    *   @param A. The tensor where the operation is applied
//    */
//    void exp_(Tensor *A);
//
//    /**
//    *   @brief Element-wise exp operation of a tensor
//    *   @param A. The tensor where the operation is applied
//    *   @return A new tnesor with the exp operation applied on A
//    */
//    Tensor* exp(Tensor *A);
//
//
//    /**
//    *   @brief Inplace element-wise floor operation
//    *   @param A. The tensor where the operation is applied
//    */
//    void floor_(Tensor *A);
//
//    /**
//    *   @brief Element-wise floor operation
//    *   @param A. The tensor where the operation is applied
//    *   @return A new tensor with the floor operation applied on A
//    */
//    Tensor* floor(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise increment operation
//    *   @param A. A tensor.
//    *   @param B. Another tensor where A+B element-wise is stored.
//    */
//    void inc_(Tensor *A,Tensor *B);
//
//    /**
//    *   @brief Inplace element-wise log operation
//    *   @param A. The tensor where the operation is applied
//    */
//    void log_(Tensor *A);
//
//    /**
//    *   @brief Element-wise log operation
//    *   @param A. The tensor where the operation is applied
//    *   @return A new tensor with the log operation applied on A
//    */
//    Tensor* log(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise log2 operation
//    *   @param A. The tensor where the operation is applied
//    */
//    void log2_(Tensor *A);
//
//    /**
//    *   @brief Element-wise log2 operation
//    *   @param A. The tensor where the operation is applied
//    *   @return A new tensor with the log2 operation applied on A
//    */
//    Tensor* log2(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise log10 operation
//    *   @param A. The tensor where the operation is applied
//    */
//    void log10_(Tensor *A);
//
//    /**
//    *   @brief Element-wise log10 operation
//    *   @param A. The tensor where the operation is applied
//    *   @return A new tensor with the log10 operation applied on A
//    */
//    Tensor* log10(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise logn operation
//    *   @param A. The tensor where the operation is applied
//    */
//    void logn_(Tensor *A,float n);
//
//    /**
//    *   @brief Element-wise logn operation
//    *   @param A. The tensor where the operation is applied
//    *   @return A new tensor with the logn operation applied on A
//    */
//    Tensor* logn(Tensor *A, float n);
//
//
//    /**
//    *   @brief Obtain the maximum value in a tensor
//    *   @param A. The tensor where the operation is applied
//    *   @return float. The maximum value in A
//    */
//    float max(Tensor *A);
//
//     /**
//    *   @brief Obtain the minimum value in a tensor
//    *   @param A. The tensor where the operation is applied
//    *   @return float. The minimum value in A
//    */
//    float min(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise mod operation.
//    *   @param A. The tensor where the operation is applied
//    *   @param v. The mod operator
//    */
//    void mod_(Tensor *A,float v);
//
//    /**
//    *   @brief Element-wise mod operation.
//    *   @param A. The tensor where the operation is applied
//    *   @param v. The mod operator
//    *   @return A tensor with A mod v
//    */
//    Tensor* mod(Tensor *A, float v);
//
//    /**
//    *   @brief Inplace multiplication operation of a tensor by a scalar.
//    *   @param A. The tensor where the operation is applied
//    *   @param v. The value to multiply by
//    */
//    void mult_(Tensor *A,float v);
//
//    /**
//    *   @brief Multiplication operation of a tensor by a scalar.
//    *   @param A. The tensor where the operation is applied
//    *   @param v. The value to multiply by
//    *   @return A tensor with A*v
//    */
//    Tensor* mult(Tensor *A, float v);
//
//    /**
//    *   @brief Inplace element-wise  multiplication operation of two 1D tensors.
//    *   @param A. The tensor where the operation is applied
//    *   @param B. Another tensor.
//    */
//    void mult_(Tensor *A, Tensor *B);
//
//    /**
//    *   @brief Element-wise multiplication operation of two 1D tensors.
//    *   @param A. A tensor.
//    *   @param B. Another tensor.
//    *   @return A tensor with A*B, element-wise
//    */
//    Tensor *mult(Tensor *A, Tensor *B);
//
//    /**
//    *   @brief Multiplication operation of two 2D tensors.
//    *   @param A. A tensor.
//    *   @param B. Another tensor.
//    *   @return A tensor with A*B
//    */
//    Tensor *mult2D(Tensor *A, Tensor *B);
//
//    /**
//    *   @brief Inplace element-wise change of sign operation.
//    *   @param A. The tensor where the operation is applied.
//    */
//    void neg_(Tensor *A);
//
//    /**
//    *   @brief Element-wise change of sign operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A tensor with -A
//    */
//    Tensor* neg(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise normalization of values in a given range.
//    *   @param A. The tensor where the operation is applied.
//    *   @param min. The lower bound of the new range
//    *   @param max. The upper bound of the new range
//    */
//    void normalize_(Tensor *A,float min, float max);
//
//    /**
//    *   @brief Inplace element-wise normalization of values in a given range.
//    *   @param A. The tensor where the operation is applied.
//    *   @param min. The lower bound of the new range
//    *   @param max. The upper bound of the new range
//    *   @return A tensor with A normalized
//    */
//    Tensor* normalize(Tensor *A, float min, float max);
//
//    /**
//    *   @brief Inplace element-wise power operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @param exp. The exponent
//    */
//    void pow_(Tensor *A,float exp);
//
//    /**
//    *   @brief Element-wise power operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @param exp. The exponent
//    *   @return A tensor with A^exp, element-wise
//    */
//    Tensor* pow(Tensor *A, float exp);
//
//    /**
//    *   @brief Inplace element-wise reciprocal operation.
//    *   @param A. The tensor where the operation is applied.
//    */
//    void reciprocal_(Tensor *A);
//
//    /**
//    *   @brief Element-wise power operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A tensor with reciprocal(A), element-wise
//    */
//    Tensor* reciprocal(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise reminder operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @param v. The real to divide A by
//    */
//    void remainder_(Tensor *A,float v);
//
//    /**
//    *   @brief Element-wise reminder operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @param v. The real to divide A by
//    *   @return A tensor with A%v
//    */
//    Tensor* remainder(Tensor *A, float v);
//
//    /**
//    *   @brief Inplace element-wise round operation.
//    *   @param A. The tensor where the operation is applied.
//    */
//    void round_(Tensor *A);
//
//    /**
//    *   @brief Element-wise round operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A tensor with A rounded
//    */
//    Tensor* round(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise inverse square root operation.
//    *   @param A. The tensor where the operation is applied.
//    */
//    void rsqrt_(Tensor *A);
//
//
//    /**
//    *   @brief Element-wise inverse square root operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A tensor with 1/sqrt(A)
//    */
//    Tensor* rsqrt(Tensor *A);
//
//
//
//    /**
//    *   @brief Inplace element-wise sigmoid operation.
//    *   @param A. The tensor where the operation is applied.
//    */
//    void sigmoid_(Tensor *A);
//
//    /**
//    *   @brief Element-wise sigmoid operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A tensor with sigmoid(A)
//    */
//    Tensor* sigmoid(Tensor *A);
//
//
//    /**
//    *   @brief Inplace element-wise sign operation. Places -1 for values lower than 0, +1 for those greater than 0 and 0 otherwise.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A tensor with A%v
//    */
//    void sign_(Tensor *A);
//
//    /**
//    *   @brief Element-wise sign operation. Places -1 for values lower than 0, +1 for those greater than 0 and 0 otherwise.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A tensor with sign(A)
//    */
//    Tensor* sign(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise sin operation.
//    *   @param A. The tensor where the operation is applied.
//    */
//    void sin_(Tensor *A);
//
//    /**
//    *   @brief Element-wise sin operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A tensor with sin(A)
//    */
//    Tensor* sin(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise sinh operation.
//    *   @param A. The tensor where the operation is applied.
//    */
//    void sinh_(Tensor *A);
//
//    /**
//    *   @brief Element-wise sinh operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A tensor with sinh(A)
//    */
//    Tensor* sinh(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise square operation. More efficient than using pow_(A, 2).
//    *   @param A. The tensor where the operation is applied.
//    */
//    void sqr_(Tensor *A);
//
//    /**
//    *   @brief Element-wise square operation. More efficient than using pow(A, 2).
//    *   @param A. The tensor where the operation is applied.
//    *   @return A tensor with sqr(A)
//    */
//    Tensor* sqr(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise square root operation.
//    *   @param A. The tensor where the operation is applied.
//    */
//    void sqrt_(Tensor *A);
//
//    /**
//    *   @brief Element-wise square operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A tensor with sqrt(A)
//    */
//    Tensor* sqrt(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise substraction operation of a tensor and a scalar.
//    *   @param A. The tensor where the operation is applied.
//    *   @param v. The value to substract to A.
//    */
//    void sub_(Tensor *A,float v);
//
//    /**
//    *   @brief Element-wise substraction operation of a tensor and a scalar.
//    *   @param A. The tensor where the operation is applied.
//    *   @param v. The value to substract to A.
//    *   @return A tensor with A-v
//    */
//    Tensor* sub(Tensor *A, float v);
//
//    /**
//    *   @brief Inplace element-wise substraction operation of two tensors.
//    *   @param A. The tensor where the operation is applied.
//    *   @param B. The tensor to substract to A.
//    */
//    void sub_(Tensor *A, Tensor *B);
//
//    /**
//    *   @brief Element-wise substraction operation of two tensors.
//    *   @param A. The tensor where the operation is applied.
//    *   @param B. The tensor to substract to A.
//    *   @return A tensor with A-B
//    */
//    Tensor *sub(Tensor *A, Tensor *B);
//
//    /**
//    *   @brief Sum all the values in a tensor.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A float. The sum of all the elements in A
//    */
//    float sum(Tensor *A);
//
//    /**
//    *   @brief  Sum the absolute value of all the values in a tensor.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A float. The sum of the absolute value of all the elements in A
//    */
//    float sum_abs(Tensor *A);
//
//
//    /**
//    *   @brief Inplace element-wise tan operation.
//    *   @param A. The tensor where the operation is applied.
//    */
//    void tan_(Tensor *A);
//
//    /**
//    *   @brief Element-wise tan operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A tensor with tan(A)
//    */
//    Tensor* tan(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise tanh operation.
//    *   @param A. The tensor where the operation is applied.
//    */
//    void tanh_(Tensor *A);
//
//    /**
//    *   @brief Element-wise tanh operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A tensor with tanh(A)
//    */
//    Tensor* tanh(Tensor *A);
//
//    /**
//    *   @brief Inplace element-wise truncate operation.
//    *   @param A. The tensor where the operation is applied.
//    */
//    void trunc_(Tensor *A);
//
//    /**
//    *   @brief Element-wise truncate operation.
//    *   @param A. The tensor where the operation is applied.
//    *   @return A tensor with trunc(A)
//    */
//    Tensor* trunc(Tensor *A);
//
//    //reductions
//    tensor reduce_mean(tensor A,vector<int> axis);
//    tensor reduce_variance(tensor A,vector<int> axis);
//
//
//
//}
//
//#endif
