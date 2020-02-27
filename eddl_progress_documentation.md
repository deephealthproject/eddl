### Development Status

| Image | Meaning |
| ------------- |------|
| ✔️ | Done |
| ❌️ | Todo |

# Welcome page


# Installation

## Installation

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Using the Conda package      | ❌️ | ✔️ |          |
| Using the Debian package     | ❌️ | ✔️ | note: not yet available|
| Using the Homebrew package   | ❌️ | ✔️ |          |
| From source with cmake       | ✔️ | ✔️ |          |
| Including EDDL in your project | ❌️ | ✔️ |          |

## Build and configuration

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| External dependencies | ✔️ |    ✔️     |          |
| Build and optimization |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Build | ✔️ | ✔️ |    |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Backend support | ✔️ | ✔️ | FPGA support note: not yet implemented |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Additional flags | ✔️ |  ✔️ |   |


## Troubleshoot


| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Segmentation fault (core dumped) | ✔️ |   | maybe needs more clarifications |
| Protobuf problems | ✔️ |      ✔️       | check use of English  <br /> maybe explain how to know if protobuf and libprotobuf are installed in standard paths|
| OpenMP  |    ✔️        |       ✔️      | check use of English |
| Import/Export Numpy files |  |       | check use of English |
| My model doesn't fit on the GPU but on X deep-learning framewok does |   | ✔️ | maybe needs more clarifications |

## FAQ


| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Is there a Python version? | ✔️ |     |          |
| Can I contribute? | ✔️ |              |          |
| Can I control the memory consumption? | ✔️ |   | maybe needs more clarifications |
| Is it faster than PyTorch/TensorFlow/etc | ✔️ |   | check use of English <br /> add link to the benchmark section |
| Is it more memory-efficient than PyTorch/TensorFlow/etc |   |   | check use of English <br /> add link to "can I control the mem. cons.?" |

# Usage

## Getting started

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| First example |   ❌️  |      ✔️       | maybe needs a more detailed explanation |
| Building with cmake | ✔️ |     ✔️      | check use of English |

## Intermediate models

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Trainig a simple MLP | ❌️|   ✔️       | image of model too small. Enable zoom on the image |
| Training a CNN | ❌️   |     ✔️        | image of model too small. Enable zoom on the image |


## Advanced models

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Training a ResNet50 |❌️ |   ✔️        | image of model too small. Enable zoom on the image |
| Training a U-Net | ❌️ |      ✔️       | image of model too small. Enable zoom on the image |


# Videotutorials

## Development

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Installing Miniconda |   |           | maybe the video should be uploaded to an official EDDL yt channel |
| Installing EDDL from source with cmake |   |   | maybe the video should be uploaded to an official EDDL yt channel |

## Showcase

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Training a simple MLP |   |          | maybe the video should be uploaded to an official EDDL yt channel |


# Layers

## Core

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Dense   |    ❌️       |      ✔️       |          |
| Embedding |  ❌️       |      ✔️       |          |
| Reshape |    ❌️       |      ✔️       |          |
| Transpose |  ❌️       |      ✔️       |          |
| Input   |    ❌️       |      ✔️       |          |
| Droput  |    ❌️       |      ✔️       |          |

## Activations

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Softmax |     ❌️      |      ✔️       |          |
| Sigmoid |     ❌️      |      ✔️       |          |
| ReLu    |     ❌️      |      ✔️       |          |
| Threshold ReLu |  ❌️  |      ✔️       |          |
| Leaky ReLu |  ❌️      |      ✔️       |          |
| ELu     |     ❌️      |      ✔️       |          |
| SeLu    |     ❌️      |      ✔️       |          |
| Exponential | ❌️      |      ✔️       |          |
| Softplus |    ❌️      |      ✔️       |          |
| Softsign |    ❌️      |      ✔️       |          |
| Linear  |     ❌️      |      ✔️       |          |
| Tanh    |     ❌️      |      ✔️       |          | 


## Data augmentation

* check use of English
* note: work in progress
* check "currently implemented"

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| RandomCrop |    ❌️    |      ✔️       |          |
| RandomCropScale | ❌️  |      ✔️       |          |
| RandomCutout |    ❌️  |      ✔️       |          |
| RandomFlip |      ❌️  |      ✔️       |          |
| RandomGrayscale | ❌️  |      ✔️       | note: not yet implemented |
| RandomHorizontalFlip | ❌️|   ✔️       |          |
| RandomRotation |  ❌️  |      ✔️       |          |
| RandomScale |     ❌️  |      ✔️       |          |
| RandomShift |     ❌️  |      ✔️       |          |
| RandomVerticalFlip |❌️|      ✔️       |          |


## Data transformation

* improve explanation
* note: work in progress
* check "currently implemented"

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Crop    |      ❌️     |      ✔️       |          |
| CenteredCrop | ❌️     |      ✔️       |          |
| ColorJitter |  ❌️     |      ✔️       | note: not yet implemented |
| CropScale |    ❌️     |      ✔️       |          |
| Cutout  |      ❌️     |      ✔️       |          |
| Flip    |      ❌️     |      ✔️       |          |
| Grayscale |    ❌️     |      ✔️       | note: not yet implemented |
| HorizontalFlip | ❌️   |      ✔️       |          |
| Pad     |      ❌️     |      ✔️       | note: not yet implemented |
| Rotate  |      ❌️     |      ✔️       |          |
| Scale   |      ❌️     |      ✔️       |          |
| Shift   |      ❌️     |      ✔️       |          |
| VerticalFlip | ❌️     |      ✔️       |          |
| Normalize |    ❌️     |      ✔️       | note: not yet implemented |


## Convolutions

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Conv2D  |     ❌️      |      ✔️       |          |
| 2D Upsampling | ❌️    |      ✔️       | note about future versions |
| Convolutional Transpose | ❌️|  ✔️     |          |


## Noise Layers

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Gaussian Noise |   ❌️ |      ✔️       |          |


## Pooling

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| MaxPooling |    ❌️    |      ✔️       |          |
| GlobalMaxPooling | ❌️ |      ✔️       |          |
| AveragePooling |   ❌️ |      ✔️       | note: not yet implemented |
| GlobalAveragePooling | ❌️|   ✔️       | note: not yet implemented |


## Normalization

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| BatchNormalization |❌️|      ✔️       |          |
| LayerNormalization |❌️|      ✔️       |          |
| GroupNormalization |❌️|      ✔️       |          |


## Merge

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Add     |     ❌️      |      ✔️       |          |
| Average |     ❌️      |      ✔️       |          |
| Concat  |     ❌️      |      ✔️       |          |
| MatMul  |     ❌️      |      ✔️       |          |
| Maximum |     ❌️      |      ✔️       |          |
| Minimum |     ❌️      |      ✔️       |          |
| Subtract |    ❌️      |      ✔️       |          |


## Generators

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Gaussian Generator |❌️|      ✔️       |          |
| Uniform Generator | ❌️|      ✔️       |          |


## Operators

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Abs     |      ❌️     |      ✔️       |          |
| Subtraction |  ❌️     |      ✔️       | three versions |
| Division |     ❌️     |      ✔️       | three versions |
| Exponent |     ❌️     |      ✔️       |          |
| Logarithm (natural) |❌️|     ✔️       |          |
| Logarithm base 2 | ❌️ |      ✔️       |          |
| Logarithm base 10 |❌️ |      ✔️       |          |
| Multiplication |   ❌️ |      ✔️       | three versions |
| Power   |      ❌️     |      ✔       | two versions |
| Sqrt    |      ❌️     |      ✔️       |          |
| Addition |     ❌️     |      ✔️       | three versions (sum) |
| Select  |      ❌️     |      ✔️       |          |
| Permute |      ❌️     |      ✔️       |          |
| ReduceMean |   ❌️     |      ✔️       |          |
| ReduceVar |    ❌️     |      ✔️       |          |
| ReduceSum |    ❌️     |      ✔️       |          |
| ReduceMax |    ❌️     |      ✔️       |          |
| ReduceMin |    ❌️     |      ✔️       |          |


## Recurrent

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| RNN     |      ❌️     |      ✔️       | note: not yet implemented |
| GRU     |      ❌️     |      ✔️       | note: not yet implemented |
| LSTM    |      ❌️     |      ✔️       | note: not yet implemented |


# Model

## Model

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Constructor |  ❌️     |      ✔️       |          |
| Build   |      ❌️     |      ✔️       |          |
| Summary |      ❌️     |      ✔️       |          |
| Plot    |      ❌️     |      ✔️       |          |
| Load    |      ❌️     |      ✔️       |          |
| Save    |      ❌️     |      ✔️       |          |
| Learning rate (on the fly) |❌️| ✔️    |          |
| Logging |      ❌️     |      ✔️       |          |
| Move to device | ❌️   |      ✔️       | two different instructions |



# Training

## Coarse training

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Fit     |      ❌️     |       ✔️      |          |


## Fine-grained training
| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| random_indices | ❌️   |       ✔️      |          |
| next_batch |     ❌️   |       ✔️      |          |
| train_batch |    ❌️   |       ✔️      |          |
| eval_batch |     ❌️   |       ✔️      |          |
| set_mode |       ❌️   |       ✔️      |          |
| reset_loss |     ❌️   |       ✔️      |          |
| forward |        ❌️   |       ✔️      |          |
| zeroGrads |      ❌️   |       ✔️      |          |
| backward |       ❌️   |       ✔️      |          |
| update  |        ❌️   |       ✔️      |          |
| print_loss |     ❌️   |       ✔️      |          |
| clamp   |        ❌️   |       ✔️      |          |
| compute_loss |   ❌️   |       ✔️      |          |
| compute_metric | ❌️   |       ✔️      |          |
| getLoss |        ❌️   |       ✔️      |          |
| newloss |        ❌️   |       ✔️      |          |
| getMetric |      ❌️   |       ✔️      |          |
| newmetric |      ❌️   |       ✔️      |          |
| detach  |        ❌️   |       ✔️      |          |


# Test & score

## Test & score

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Evaluate model | ❌️   |       ✔️      |          |


# Bundle

## Losses

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Mean Squared Error |❌️|      ✔️       |          |
| Cross-Entropy |     ❌️|      ✔️       |          |
| Soft Cross-Entropy |❌️|      ✔️       |          |


## Metrics

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Mean Squared Error |❌️|       ✔️      |          |
| Categorical Accuracy |❌️|     ✔️      |          |
| Mean Absolute Error |❌️|      ✔️      |          |
| Mean Relative Error |❌️|      ✔️      |          |


## Regularizers

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| L1      |      ❌️     |       ✔️      |          | 
| L2      |      ❌️     |       ✔️      |          |
| L1L2    |      ❌️     |       ✔️      |          |


## Initializers

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| GlorotNormal |    ❌️  |       ✔️      |          |
| GlorotUniform |   ❌️  |       ✔️      |          |
| RandomNormal |    ❌️  |       ✔️      |          |
| RandomUniform |   ❌️  |       ✔️      |          |
| Constant |        ❌️  |       ✔️      |          |


## Optimizers

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Adadelta |     ❌️     |       ✔️      |          |
| Adam    |      ❌️     |       ✔️      |          |
| Adagrad |      ❌️     |       ✔️      |          |
| Adamax  |      ❌️     |       ✔️      |          |
| Nadam   |      ❌️     |       ✔️      |          |
| RMSProp |      ❌️     |       ✔️      |          |
| SGD (Stochastic Gradient Descent) |❌️| ✔️ |      |


# Computing services

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| CPU     |      ✔️      |       ✔️      | check use of English |
| GPU     |      ✔️      |       ✔️      | check use of English |
| FPGA    |      ✔️      |       ✔️      | note: not yet implemented |
| COMPSS  |      ✔️      |       ✔️      | note: not yet implemented |


# Datasets

## Classification

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| MNIST   |      ❌️     |       ✔️      |          |
| CIFAR   |      ❌️     |       ✔️      |          |

## Segmentation

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| DRIVE   |      ❌️     |       ✔️      |          |


# Micellaneous ❌️

empty ❌

# Tensor

## Creation Routines

* note: section in progress

### Constructor

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Tensor  |     ❌️      |       ✔️      |          |

### Ones and zeros

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| zeros   |      ✔️      |       ✔️      |          |
| ones    |      ✔️      |       ✔️      |          |
| full    |      ✔️      |       ✔️      |          |
| eye     |      ❌️     |       ✔️      |          |
| identity |     ✔️      |       ✔️      |          |

### From existing data

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| clone   |      ❌️     |       ✔️      |          |
| reallocate |   ❌️     |       ✔️      |          |
| copy    |      ❌️     |       ✔️      |          |


### Numerical ranges

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| arange  |      ❌️     |       ✔️      |          |
| range   |      ❌️     |       ✔️      |          |
| linspace |     ❌️     |       ✔️      |          |
| logspace |     ❌️     |       ✔️      |          |
| geomspace |    ❌️     |       ✔️      |          |

### Random

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| randu   |      ❌️     |       ✔️      |          |
| randn   |      ❌️     |       ✔️      |          |

### Build matrices

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| diag    |      ❌️     |       ✔️      |          |


## Manipulation

* note: section in progress 

### Constructor

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Tensor  |      ❌️     |       ✔️      |          |

### Changing array shape

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| reshape |      ❌️     |       ✔️      |          |
| flatten |      ❌️     |       ✔️      |          |

### Transpose-like operations

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| permute |      ❌️     |       ✔️      |          |
| moveaxis |     ❌️     |       ✔️      |          |
| swapaxis |     ❌️     |       ✔️      |          |

### Changing number of dimensions

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| squeeze |      ❌️     |       ✔️      |          |
| unsqueeze |    ❌️     |       ✔️      |          |

### Joining arrays

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| concat  |      ❌️     |       ✔️      |          |

### Rearranging elements and transformations

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| shift   |      ❌️     |       ✔️      |          |
| rotate  |      ❌️     |       ✔️      |          |
| scale   |      ❌️     |       ✔️      |          |
| flip    |      ❌️     |       ✔️      |          |
| crop    |      ❌️     |       ✔️      |          |
| crop_scale |   ❌️     |       ✔️      |          |
| cutout  |      ❌️     |       ✔️      |          |
| shift_random | ❌️     |       ✔️      |          |
| rotate_random |❌️     |       ✔️      |          |
| scale_random | ❌️     |       ✔️      |          |
| flip_random |  ❌️     |       ✔️      |          |
| crop_random |  ❌️     |       ✔️      |          |
| crop_scale_random| ❌️ |       ✔️      |          |
| cutout_random |❌️     |       ✔️      |          |


## Binary operations ❌️

* note: section in progress 

## Indexing routines

* note: section in progress

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Generating index arrays |❌️|    ❌️   |          |
| Indexing-like operations | 
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;select |❌️| ✔️ |   |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;set_select |❌️| ✔️ |   |


## Input/Output Operations

* note: section in progress

### Input
| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| loadfs  |      ❌️     | ✔️            |          |
| load    |      ❌️     | ✔️            |          | 
| load_from_txt |❌️     | ✔️            |          |

### Output

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| savefs  |      ❌️     |       ✔️      |          |
| save    |      ❌️     |       ✔️      |          |
| save2txt |     ❌️     |       ✔️      |          |


## Linar algebra

* note: section in progress

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Matrix and vector products | ❌️| ❌️  | just 'interpolate' |
| Norms and other numbers | ❌️|    ❌️  |          |
| Solving equations and inverting matrices |❌️ |❌️ |   |


## Logic functions

* note: section in progress

### Truth value testing 

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| all     |      ❌️     |       ✔️      |          |
| any     |      ❌️     |       ✔️      |          |

### Array contents

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| isfinite |     ❌️     |       ✔️      |          |
| isinf   |      ❌️     |       ✔️      |          |
| isnan   |      ❌️     |       ✔️      |          |
| isneginf |     ❌️     |       ✔️      |          |
| isposinf |     ❌️     |       ✔️      |          |

### Logical operations

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| logical_and |   ❌️    |       ✔️      |          |
| logical_or |    ❌️    |       ✔️      |          |
| logical_not |   ❌️    |       ✔️      |          |
| logical_xor |   ❌️    |       ✔️      |          |

### Comparison

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| allclose |     ❌️     |       ✔️      |          |
| isclose |      ❌️     |       ✔️      |          |
| greater |      ❌️     |       ✔️      |          |
| greater_equal |❌️     |       ✔️      |          |
| less    |      ❌️     |       ✔️      |          |
| less_equal |   ❌️     |       ✔️      |          |
| equal   |      ❌️     |       ✔️      |          |
| not_equal |    ❌️     |       ✔️      |          |

## Masked array operations ❌️

* note: section in progress

## Mathematical functions 

* note: section in progress

### Element-wise

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| abs     |      ❌️     |       ✔️      |          |
| acos    |      ❌️     |       ✔️      |          |
| add     |      ❌️     |       ✔️      | three instructions |
| asin    |      ❌️     |       ✔️      |          |
| atan    |      ❌️     |       ✔️      |          |
| ceil    |      ❌️     |       ✔️      |          |
| clamp   |      ❌️     |       ✔️      |          |
| clampmax |     ❌️     |       ✔️      |          |
| clampmin |     ❌️     |       ✔️      |          |
| cos     |      ❌️     |       ✔️      |          |
| cosh    |      ❌️     |       ✔️      |          |
| div     |      ❌️     |       ✔️      | two instructions | 
| exp     |      ❌️     |       ✔️      |          |
| floor   |      ❌️     |       ✔️      |          |
| log     |      ❌️     |       ✔️      |          |
| log2    |      ❌️     |       ✔️      |          |
| log10   |      ❌️     |       ✔️      |          |
| logn    |      ❌️     |       ✔️      |          |
| mod     |      ❌️     |       ✔️      |          |
| mult    |      ❌️     |       ✔️      | two instructions |
| neg     |      ❌️     |       ✔️      |          |
| pow     |      ❌️     |       ✔️      |          |
| reciprocal |   ❌️     |       ✔️      |          |
| remainder |    ❌️     |       ✔️      |          |
| round   |      ❌️     |       ✔️      |          |
| rsqrt   |      ❌️     |       ✔️      |          |
| sigmoid |      ❌️     |       ✔️      |          |
| sign    |      ❌️     |       ✔️      | two instructions |
| sin     |      ❌️     |       ✔️      |          |
| sinh    |      ❌️     |       ✔️      |          |
| sqr     |      ❌️     |       ✔️      |          |
| sqrt    |      ❌️     |       ✔️      |          |
| sub     |      ❌️     |       ✔️      |          |
| sum     |      ❌️     |       ✔️      | three different instructions |
| tan     |      ❌️     |       ✔️      |          |
| tanh    |      ❌️     |       ✔️      |          |
| trunc   |      ❌️     |       ✔️      |          |

### Reductions

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| max     |      ❌️     |       ✔️      |          |
| min     |      ❌️     |       ✔️      |          |


## Miscellaneous

* note: section in progress

### Functions

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| toCPU   |     ❌️      |       ✔️      |          |
| toGPU   |     ❌️      |       ✔️      |          |
| isCPU   |     ❌️      |       ✔️      |          |
| isGPU   |     ❌️      |       ✔️      |          |
| isFPGA  |     ❌️      |       ✔️      |          |
| isSquared |   ❌️      |       ✔️      |          |
| copy    |     ❌️      |       ✔️      |          |
| clone   |     ❌️      |       ✔️      |          |
| info    |     ❌️      |       ✔️      |          |
| print   |     ❌️      |       ✔️      |          |
| valid_indices | ❌️    |       ✔️      |          |
| get_address_rowmajor |❌️|     ✔️      |          |