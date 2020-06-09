# Tensor Routines

Numpy-like operations over a raw-tensor object

##### Legend: Development status

| Image | Meaning |
| ------------- |------|
| ✔️ | Done |
| ❌ | Todo |

---
## Array creation routines

### Ones and zeros

---


| Functionality | CPU  | GPU  | Comments |
| ------------- | ---- | ---- | -------- |
| empty         | ✔️    | ✔️    |   Returns a tensor filled with uninitialized data.       |
| empty_like         | ✔️    | ✔ ️    |   Returns a tensor filled with uninitialized data, with the same size as the input tensor       |
| eye           | ✔️    | ✔️    |   Return a 2-D array with ones on the diagonal and zeros elsewhere.       |
| identity      | ✔️    | ✔️    |    Return the identity array (eye with offset=0).      |
| ones          | ✔️    | ✔️    |    Return a new array of given shape and type, filled with ones.      |
| ones_like         | ✔️    | ✔    |     Returns a tensor filled with the scalar value 1, with the same size as the input tensor     |
| zeros         | ✔️    | ✔️    |     Return a new array of given shape and type, filled with zeros.     |
| zeros_like         | ✔️    | ✔    |     Returns a tensor filled with the scalar value 0, with the same size as the input tensor     |
| full          | ✔️    | ✔️    |   Return a new array of given shape and type, filled with "value".       |
| full_like         | ✔️    | ✔    |     Returns a tensor filled with the given scalar value, with the same size as the input tensor     |


### From existing data

| Functionality | CPU  | GPU  | Comments                                                     |
| ------------- | ---- | ---- | ------------------------------------------------------------ |
| Tensor         |  ✔️    | ✔    | Constructs a tensor with data                           |
| clone         | ✔️    | ✔️    | Creates an identical (but different) tensor from another                           |
| copy          | ✔️ | ✔️ |  Copy data from Tensor A to B |


### Numerical ranges

| Functionality | CPU  | GPU  | Comments |
| ------------- | ---- | ---- | -------- |
| arange        | ✔️    | ✔️    | Return evenly spaced values within a given interval `[0, n)`     |
| range         | ✔️    | ✔️    | Return evenly spaced values within a given interval. `[0, n]`      |
| linspace      | ✔️    | ✔️    | Return evenly spaced numbers over a specified interval.         |
| logspace      | ✔️    | ✔️    | Return numbers spaced evenly on a log scale.         |
| geomspace      | ✔️    | ✔️    | Return numbers spaced evenly on a log scale (a geometric progression).         |


### Random


| Functionality | CPU  | GPU  | Comments |
| ------------- | ---- | ---- | -------- |
| randu        | ✔️    | ✔️     | 	Return a uniform random matrix with given shape.   |
| randn        | ✔️    | ✔️    | 	Return a normal random matrix with data from the "standard normal" distribution.     |
| rand_uniform        | ✔️    | ✔️    |     |
| rand_signed_uniform        | ✔️    | ✔️    |     |
| rand_normal        | ✔️    | ✔️    |     |
| rand_binary        | ✔️    | ✔️    |     |


### Building matrices

| Functionality | CPU  | GPU  | Comments |
| ------------- | ---- | ---- | -------- |
| diag      | ✔️    | ✔️    |  Extract a diagonal or construct a diagonal array.        |
| tri      | ❌    | ❌    | An array with ones at and below the given diagonal and zeros elsewhere.         |


## Array manipulation routines

### Changing array shape

| Functionality | CPU  | GPU  | Comments |
| ------------- | ---- | ---- | -------- |
| reshape      | ✔️    | ✔️    |  Gives a new shape to an array without changing its data.        |
| flatten      | ✔️    | ✔️    |  Return a copy of the array collapsed into one dimension.       |


### Transpose-like operations

| Functionality | CPU  | GPU  | Comments |
| ------------- | ---- | ---- | -------- |
| moveaxis      | ✔️    | ✔️    |  Move axes of an array to new positions.  `(1, 3): [0,1,2,3] => [0,2,3,1]`     |
| swapaxes      | ✔️    | ✔️    |  Interchange two axes of an array.  `(1, 3): [0,1,2,3] => [0,3,2,1]`      |
| permute      | ✔️    | ✔️    |  Permute the dimensions of an array.       |


### Changing number of dimensions

| Functionality | CPU  | GPU  | Comments |
| ------------- | ---- | ---- | -------- |
| squeeze      | ✔️    | ✔️    |  Remove single-dimensional entries from the shape of an array. `[3, 4, 1, 7] => [3,4,7]`     |
| unsqueeze      | ✔️    | ✔️    |  Expand the shape of an array.  `[3, 4, 7] => [1, 3, 4, 7]`      |


### Joining arrays

| Functionality | CPU  | GPU  | Comments |
| ------------- | ---- | ---- | -------- |
| concatenate      |  ✔️    | ✔️      |  Join a sequence of arrays along an existing axis.    |
| stack      | ❌    | ❌    |  Join a sequence of arrays along a new axis.    |


### Splitting arrays

| Functionality | CPU  | GPU  | Comments |
| ------------- | ---- | ---- | -------- |
| split      | ❌    | ❌    |  Split an array into multiple sub-arrays.   |

### Tiling arrays

| Functionality | CPU  | GPU  | Comments |
| ------------- | ---- | ---- | -------- |
| tile        | ❌    | ❌    |  	Construct an array by repeating A the number of times given by reps.   |
| repeat      | ❌    | ❌    |  Repeat elements of an array.  |


### Adding and removing elements

| Functionality | CPU  | GPU  | Comments |
| ------------- | ---- | ---- | -------- |
| delete      | ❌    | ❌    |  Return a new array with sub-arrays along an axis deleted.  |
| insert      | ❌    | ❌    |  Insert values along the given axis before the given indices. |
| append      | ❌    | ❌    |  Append values to the end of an array. |
| trim_zeros      | ❌    | ❌    |  Trim the leading and/or trailing zeros from a 1-D array or sequence.  |
| unique      | ❌    | ❌    |  Find the unique elements of an array.  |


### Rearranging elements and transformations

| Functionality | CPU  | GPU  | Comments |
| ------------- | ---- | ---- | -------- |
| shift      | ✔️    | ✔️    |  	   |
| rotate      | ✔️    | ✔️    |  	   |
| scale      | ✔️    | ✔️    |  	   |
| flip      | ✔️    | ✔️    |  	   |
| crop      | ✔️    | ✔️    |  	   |
| crop_scale      | ✔️    | ✔️    |  	   |
| cutout      | ✔️    | ✔️    |  	   |
| shift_random      | ✔️    | ✔️    |  	   |
| rotate_random      | ✔️    | ✔️    |  	   |
| scale_random      | ✔️    | ✔️    |  	   |
| flip_random      | ✔️    | ✔️    |  	   |
| crop_random      | ✔️    | ✔️    |  	   |
| crop_scale_random      | ✔️    | ✔️    |  	   |
| cutout_random      | ✔️    | ✔️    |  	   |



## Binary operations

Not yet...

## Indexing routines

### Generating index arrays

| Functionality | CPU  | GPU  | Comments                                                     |
| ------------- | ---- | ---- | ------------------------------------------------------------ |
| nonzero          | ✔️    | ❌     | Return the indices of the elements that are non-zero.                                      |
| where          | ✔️    | ✔️    | Return elements, either from x or y, depending on condition.                                      |
| mask_indices          | ❌    | ❌️      | Return the indices to access (n, n) arrays, given a masking function.                                     |


### Indexing-like operations

| Functionality | CPU  | GPU  | Comments                                                     |
| ------------- | ---- | ---- | ------------------------------------------------------------ |
| select          | ✔️    | ✔️    | Returns an array with the selected indices. `Tensor::select(k); k=vector of strings ({"0", ":5", ":", "3:6"})`. _TODO: Accept masks_   |
| set_select          | ✔️    | ✔️    | Sets the elements in the array using the selected indices        `Tensor::set_select({"0", ":5", ":", "3:6"}, k); //k=float or Tensor                           |
| index_select          | ❌    | ❌️    | Returns a new tensor which indexes the input tensor along dimension dim using the entries in index                          |
| masked_select          | ❌    | ❌️    | Returns a new 1-D tensor which indexes the input tensor according to the boolean mask                           |
| take          | ❌    | ❌️    | Returns a new tensor with the elements of input at the given indices. The input tensor is treated as if it were viewed as a 1-D tensor.                          |


## Input and output

### Input

| Functionality | CPU  | GPU  | Comments                                                     |
| ------------- | ---- | ---- | ------------------------------------------------------------ |
| load          | ✔️    | -    | Images: jpg, png, bmp, hdr, psd, tga, gif, pic, pgm, ppm<br />Numpy (removed): npy, npz<br />Text (removed): csv, tsv, txt,...<br />Other: onnx, bin |
| memmap | ❌    | -    |  Create a memory-map to an array stored in a binary file on disk.                                                            |

### Output

| Functionality | CPU  | GPU  | Comments                                                     |
| ------------- | ---- | ---- | ------------------------------------------------------------ |
| save          | ✔️    | -    | Images: jpg, png, bmp, hdr, psd, tga, gif, pic, pgm, ppm<br />Numpy (removed): npy, npz<br />Text: csv, tsv, txt,...<br />Other: bin                                    |
| save2txt      | ✔️    | -     |                                                              |


## Linear algebra

### Matrix and vector products

| Functionality | CPU  | GPU  | Comments                                                     |
| ------------- | ---- | ---- | ------------------------------------------------------------ |
| dot      | ❌    |  ❌    |  Dot product of two arrays.                                                            |
| inner      | ❌    |   ❌   |    Inner product of two arrays.                                                          |
| outer      | ❌    |   ❌   |     Compute the outer product of two vectors.                                                         |
| matmul      | ❌    |   ❌   |           Matrix product of two arrays.                                                   |
| tensordot      | ❌    |   ❌   |     Compute tensor dot product along specified axes for arrays >= 1-D.                                                         |
| eigensum      | ❌    |   ❌   |   Evaluates the Einstein summation convention on the operands                                                           |
| interpolate | ✔️ | ✔️ |  Interpolate two tensors: `c*A + (1-c)*B` |


### Norms and other numbers

| Functionality | CPU  | GPU  | Comments                                                     |
| ------------- | ---- | ---- | ------------------------------------------------------------ |
| norm      | ✔️    |  ✔️    |  Matrix or vector norm.                                                 |
| det      | ❌    |   ❌   |   Compute the determinant of an array.                                                        |
| trace      | ✔️    |  ✔️    |   Return the sum along diagonals of the array.                                                 |

### Solving equations and inverting matrices

| Functionality | CPU  | GPU  | Comments                                                     |
| ------------- | ---- | ---- | ------------------------------------------------------------ |
| inv      | ❌    |  ❌    |  Compute the (multiplicative) inverse of a matrix.                                      |


## Logic functions

### Truth value testing

| Functionality | CPU  | GPU  | Comments                                                     |
| ------------- | ---- | ---- | ------------------------------------------------------------ |
| all      | ✔️    |  ✔️    |  Test whether all array elements along a given axis evaluate to True.                        |
| any      | ✔️    |  ✔️    |  Test whether any array element along a given axis evaluates to True                        |


### Array contents

| Functionality | CPU  | GPU  | Comments                                                     |
| ------------- | ---- | ---- | ------------------------------------------------------------ |
| isfinite      | ✔️    |  ✔️    |  Test element-wise for finiteness (not infinity or not Not a Number).                |
| isinf         | ✔️    |  ✔️    |                  Test element-wise for positive or negative infinity.   |
| isnan         | ✔️    |  ✔️    |                   Test element-wise for NaN and return result as a boolean array.   |
| isneginf      | ✔️    |  ✔️    |                   Test element-wise for negative infinity, return result as bool array.   |
| isposinf      | ✔️    |  ✔️    |                   	Test element-wise for positive infinity, return result as bool array.   |


### Logical operations

| Functionality | CPU  | GPU  | Comments                                                     |
| ------------- | ---- | ---- | ------------------------------------------------------------ |
| logical_and      | ✔️    |  ✔️    |  Compute the truth value of x1 AND x2 element-wise. |
| logical_or      | ✔️    |  ✔️    |  Compute the truth value of x1 OR x2 element-wise.
| logical_not      | ✔️    |  ✔️    | Compute the truth value of NOT x element-wise.  |
| logical_xor      | ✔️    |  ✔️    |  Compute the truth value of x1 XOR x2, element-wise. |


### Comparison

#### Boolean

| Functionality | CPU  | GPU  | Comments                                                     |
| ------------- | ---- | ---- | ------------------------------------------------------------ |
| allclose      | ✔️    |  ✔️    | Returns True if two arrays are element-wise equal within a tolerance.  |
| isclose      | ✔️    |  ✔️    | Returns a boolean array where two arrays are element-wise equal within a tolerance.  |
| greater      | ✔️    |  ✔️    | Return the truth value of (x1 > x2); Tensor-Tensor, Tensor-float  |
| greater_equal      | ✔️    |  ✔️    | Return the truth value of (x1 >= x2); Tensor-Tensor, Tensor-float  |
| less      | ✔️    |  ✔️    | Return the truth value of (x1 < x2) element-wise; Tensor-Tensor, Tensor-float  |
| less_equal      | ✔️    |  ✔️    | Return the truth value of (x1 =< x2) element-wise; Tensor-Tensor, Tensor-float  |
| equal      | ✔️    |  ✔️    | Return (x1 == x2) element-wise; Tensor-Tensor, Tensor-float  |
| not_equal      | ✔️    |  ✔️    | Return (x1 != x2) element-wise; Tensor-Tensor, Tensor-float  |


#### Indices

| argsort      | ❌    |  ❌ ️    | Returns the indices that sort a tensor along a given dimension in ascending order by value.  |
| kthvalue      | ❌    |  ❌ ️    | Returns a namedtuple (values, indices) where values is the k th smallest element of each row of the input tensor in the given dimension dim  |
| sort      | ❌    |  ❌ ️    | Sorts the elements of the input tensor along a given dimension in ascending order by value.  |
| topk      | ❌    |  ❌ ️    | Returns the k largest elements of the given input tensor along a given dimension.  |


## Masked array operations

> To do...

## Mathematical functions

### Point-wise

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| abs | ✔️ | ✔️ |         |
| acos | ✔️ | ✔️ |         |
| add | ✔️ | ✔️ |         |
| asin | ✔️ | ✔️ |         |
| atan | ✔️ | ✔️ |         |
| ceil | ✔️ | ✔️ |         |
| clamp | ✔️ | ✔️ |         |
| clampmax | ✔️ | ✔️ |         |
| clampmin | ✔️ | ✔️ |         |
| cos | ✔️ | ✔️ |         |
| cosh | ✔️ | ✔️ |         |
| div | ✔️ | ✔️ |         |
| exp | ✔️ | ✔️ |         |
| floor | ✔️ | ✔️ |         |
| log | ✔️ | ✔️ |         |
| log2 | ✔️ | ✔️ |         |
| log10 | ✔️ | ✔️ |         |
| logn | ✔️ | ✔️ |         |
| max | ✔️ | ✔️ | Not reduced         |
| mean | ✔️ | ✔️ | Not reduced         |
| median* | ❌ | ❌ | Not reduced         |
| min | ✔️ | ✔️ | Not reduced         |
| mod | ✔️ | ✔️ |         |
| mode | ❌ | ❌ | Not reduced         |
| mult | ✔️ | ✔️ |         |
| neg | ✔️ | ✔️ |         |
| normalize | ✔️ | ✔️ | Not reduced         |
| pow | ✔️ | ✔️ |         |
| powb | ✔️ | ✔️ |         |
| reciprocal | ✔️ | ✔️ |         |
| remainder | ✔️ | ✔️ |         |
| round | ✔️ | ✔️ |         |
| rsqrt | ✔️ | ✔️ |         |
| sigmoid | ✔️ | ✔️ |         |
| sign | ✔️ | ✔️ |         |
| sin | ✔️ | ✔️ |         |
| sinh | ✔️ | ✔️ |         |
| sqr | ✔️ | ✔️ |         |
| sqrt | ✔️ | ✔️ |         |
| std* | ❌ | ❌ | Not reduced         |
| sub | ✔️ | ✔️ |         |
| sum | ✔️ | ✔️ | Not reduced         |
| sum_abs | ✔️ | ✔️ | Not reduced         |
| tan | ✔️ | ✔️ |         |
| tanh | ✔️ | ✔️ |         |
| trunc | ✔️ | ✔️ |         |
| var* | ❌ | ❌ | Not reduced |


### Element-wise

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| add | ✔️ | ✔️ | Tensor-Tensor, Tensor-float
| div | ✔️ | ✔️ | Tensor-Tensor, Tensor-float
| mult | ✔️ | ✔️ | Tensor-Tensor, Tensor-float
| sub | ✔️ | ✔️ | Tensor-Tensor, Tensor-float
| interpolate | ✔️ | ✔️ | Tensor-Tensor
| maximum | ✔️ | ✔️ | Tensor-Tensor, Tensor-float
| minimum | ✔️ | ✔️ | Tensor-Tensor, Tensor-float


### Reductions

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| argmax | ❌ | ❌ |
| argmin | ❌ | ❌ |
| cumprod | ❌ | ❌ |
| cumsum | ❌ | ❌ |
| max | ✔️ | ✔️ |
| min | ✔️ | ✔️ |
| mean | ✔️ | ✔️ |
| median | ❌ | ❌ |
| mode | ❌ | ❌ |
| norm | ❌ | ❌ |
| prod | ❌ | ❌ |
| std | ❌ | ❌ |
| sum | ❌ | ❌ |
| unique | ❌ | ❌ |
| var | ❌ | ❌ |




## Miscellaneous

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| toCPU | ✔️ | ✔️ | Clone a tensor to the CPU |
| toGPU | ✔️ | ✔️ | Clone a tensor to the GPU |
| isCPU | ✔️ | ✔️ | Check if the tensor if in CPU |
| isGPU | ✔️ | ✔️ | Check if the tensor if in GPU |
| isFPGA | - | - |  Check if the tensor if in FPGA |
| isSquared | ✔️ | ✔️ | Check if all dimensions in the tensors are the same |
| copy | ✔️ | ✔️ |  Copy data from Tensor A to B |
| clone | ✔️ | ✔️ | Clone a tensor (same device) |
| info | ✔️ | ✔️ | Print shape, device and size information |
| print | ✔️ | ✔️ | Prints the tensor values |
| numel |  ✔️ | ✔️ | Returns the total number of elements in the input tensor. |

