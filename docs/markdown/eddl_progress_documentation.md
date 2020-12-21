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
| Using the Conda package      | ✔ | ✔️ |          |
| Using the Homebrew package   | ✔️ | ✔️ |          |
| From source with cmake       | ✔️ | ✔️ |          |
| Including EDDL in your project | ✔️ | ✔️ |          |

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
| Compilation |  ✔️      |       ✔️      |          |
| Memory  |      ✔️      |       ✔️      |          |
| Protobuf |     ✔️      |       ✔️      |          |
| CUDA    |      ✔️      |       ✔️      |          |
| IDEs    |      ✔️      |       ✔️      |          |

## FAQ


| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Python library | ✔️    |              |          |
| Contributions  | ✔️    |              |          |
| Performance |    ✔️    |              |          |

# Usage

## Getting started

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| First example |   ✔️   |      ✔️       |          |
| Building with cmake | ✔️ |     ✔️      |          |

## Intermediate models

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Trainig a simple MLP | ✔️ |   ✔️       |          |
| Training a CNN | ✔️   |      ✔️        |          |


## Advanced models

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Training a ResNet50 | ✔️ |   ✔️        |          |
| Training a U-Net | ✔️  |      ✔️       |          |


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
| Dense   |     ✔️       |      ✔️       |          |
| Embedding |   ✔️       |      ✔️       |          |
| Reshape |     ✔️       |      ✔️       |          |
| Flatten |     ✔️       |      ✔️       |          |
| Input   |     ✔️       |      ✔️       |          |
| Droput  |     ✔️       |      ✔️       |          |
| Select  |     ✔️       |      ✔️       |          |
| Permute |     ✔️       |      ✔️       |          |
| Transpose  |  ✔️       |      ✔️       |          |

## Activations

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Softmax |      ✔️      |      ✔️       |          |
| Sigmoid |      ✔️      |      ✔️       |          |
| ReLu    |      ✔️      |      ✔️       |          |
| Threshold ReLu |  ✔️   |      ✔️       |          |
| Leaky ReLu |  ✔️       |      ✔️       |          |
| ELu     |     ✔️       |      ✔️       |          |
| SeLu    |     ✔️       |      ✔️       |          |
| Exponential | ✔️       |      ✔️       |          |
| Softplus |    ✔️       |      ✔️       |          |
| Softsign |    ✔️       |      ✔️       |          |
| Linear  |     ✔️       |      ✔️       |          |
| Tanh    |      ✔️      |      ✔️       |          |


## Data augmentation

* note: work in progress
* check "currently implemented"

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| RandomAffine  | ✔️     |      ❌️      | note: not implemented yet |
| RandomCrop |    ✔️     |      ✔️       |          |
| RandomCropScale | ✔️   |      ✔️       |          |
| RandomCutout |    ✔️   |      ✔️       |          |
| RandomFlip |      ✔️   |      ✔️       |          |
| RandomGrayscale | ✔️   |      ❌️      | note: not yet implemented |
| RandomHorizontalFlip | ✔️ |   ✔️       |          |
| RandomRotation |  ✔️   |      ✔️       |          |
| RandomScale |     ✔️   |      ✔️       |          |
| RandomShift |     ✔️   |      ✔️       |          |
| RandomVerticalFlip |✔️ |      ✔️       |          |


## Data transformation

* improve explanation
* note: work in progress
* check "currently implemented"

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Affine  |      ✔️      |      ❌️      | note: not yet implemented |
| Crop    |      ✔️      |      ✔️       |          |
| CenteredCrop | ✔️      |      ✔️       |          |
| ColorJitter |  ✔️      |      ❌️      | note: not yet implemented |
| CropScale |    ✔️      |      ✔️       |          |
| Cutout  |      ✔️      |      ✔️       |          |
| Flip    |      ✔️      |      ✔️       |          |
| Grayscale |    ✔️      |      ❌️      | note: not yet implemented |
| HorizontalFlip | ✔️    |      ✔️       |          |
| Pad     |      ✔️      |      ❌️      | note: not yet implemented |
| Rotate  |      ✔️      |      ✔️       |          |
| Scale   |      ✔️      |      ✔️       |          |
| Shift   |      ✔️      |      ❌️      |          |
| VerticalFlip | ✔️      |      ✔️       |          |
| Normalize |    ✔️      |      ❌️      | note: not yet implemented |


## Convolutions

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Conv1D  |     ✔️       |      ✔️       |          |
| Conv2D  |     ✔️       |      ✔️       |          |
| Pointwise |   ✔️       |      ✔️       |          |
| 2D Upsampling | ✔️     |      ✔️       | note about future versions |
| Convolutional Transpose | ✔️ |  ❌️    | note: not implemented yet  |


## Noise Layers

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Gaussian Noise |   ✔️  |      ✔️       |          |


## Pooling

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| MaxPooling1D |  ✔️     |      ✔️       |          |
| MaxPooling |    ✔️     |      ✔️       |          |
| GlobalMaxPooling | ✔️  |      ✔️       |          |
| AveragePooling |   ✔️  |      ✔️       |          |
| GlobalAveragePooling | ✔️ |   ✔️       |          |


## Normalization

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| BatchNormalization |✔️ |      ✔️       |          |
| LayerNormalization |✔️ |      ✔️       |          |
| GroupNormalization |✔️ |      ✔️       |          |


## Merge

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Add     |     ✔️       |      ✔️       |          |
| Average |     ✔️       |      ✔️       |          |
| Concat  |     ✔️       |      ✔️       |          |
| MatMul  |     ❌️      |      ✔️       |          |
| Maximum |     ✔️       |      ✔️       |          |
| Minimum |     ✔️       |      ✔️       |          |
| Subtract |    ✔️       |      ✔️       |          |


## Generators

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Gaussian Generator |❌️|      ✔️       | needs comments in the .h |
| Uniform Generator | ❌️|      ❌️      | needs comments in the .h |


## Operators

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Abs     |       ✔️     |      ✔️       |          |
| Subtraction |   ✔️     |      ✔️       | three versions <br/> not every version has its explanation in the .h |
| Division |      ✔️     |      ✔️       | three versions <br/> not every version has its explanation in the .h |
| Exponent |      ✔️     |      ✔️       |          |
| Logarithm (natural) | ✔️|     ✔️       |          |
| Logarithm base 2 |  ✔️ |      ✔️       |          |
| Logarithm base 10 | ✔️ |      ✔️       |          |
| Multiplication |    ✔️ |      ✔️       | three versions <br/> not every version has its explanation in the .h |
| Power   |       ✔️     |      ❌️      |          |
| Sqrt    |       ✔️     |      ✔️       |          |
| Addition |      ✔️     |      ✔️       | three versions (sum) <br/> not every version has its explanation in the .h |

## Reduction Layers 

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| ReduceMean |   ❌️     |      ✔️       | needs comments in the .h |
| ReduceVar |    ❌️     |      ❌️      | needs comments in the .h |
| ReduceSum |    ❌️     |      ✔️       | needs comments in the .h |
| ReduceMax |    ❌️     |      ✔️       | needs comments in the .h |
| ReduceMin |    ❌️     |      ✔️       | needs comments in the .h |


## Recurrent

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| RNN     |      ✔️      |      ✔️       |          |
| GRU     |      ❌️     |      ❌️      | note: not yet implemented <br/>write comments in the .h |
| LSTM    |      ✔️      |      ✔️       |          |


# Model

## Model

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Constructor |  ✔️      |      ✔️       |          |
| Build   |      ✔️      |      ✔️       | two different instructions |
| Summary |      ✔️      |      ✔️       |          |
| Plot    |      ✔️      |      ✔️       |          |
| Load    |      ✔️      |      ✔️       |          |
| Save    |      ✔️      |      ✔️       |          |
| Learning rate (on the fly) |✔️ | ✔️    |          |
| Logging |      ✔️      |      ✔️       |          |
| toCPU   |      ✔️      |      ✔️       |          |
| toGPU   |      ❌️     |      ❌️      | many versions |

## ONNX

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Save to file |  ✔️     |      ✔️       |          |
| Import from file |  ✔️ |      ✔️       |          |


# Training

## Coarse training

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Fit     |      ✔️      |       ✔️      |          |


## Fine-grained training
| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| random_indices |  ✔️   |       ✔️      |          |
| next_batch |      ✔️   |       ✔️      |          |
| train_batch |     ✔️   |       ✔️      | two different instructions |
| eval_batch |      ✔️   |       ✔️      | two different instructions |
| set_mode |        ✔️   |       ❌️     |          |
| reset_loss |      ✔️   |       ✔️      |          |
| forward |         ✔️   |       ❌️     | 4 different instructions |
| zeroGrads |       ✔️   |       ✔️      |          |
| backward |        ✔️   |       ❌️     | 3 different instructions |
| update  |         ✔️   |       ✔️      |          |
| print_loss |      ✔️   |       ✔️      |          |
| clamp   |         ✔️   |       ❌️     |          |
| compute_loss |    ✔️   |       ✔️      |          |
| compute_metric |  ✔️   |       ❌️     |          |
| getLoss |         ✔️   |       ❌️     |          |
| newloss |         ✔️   |       ✔️      | 2 different instructions |
| getMetric |       ✔️   |       ❌️     |          |
| newmetric |       ✔️   |       ❌️     | 2 different instructions |
| detach  |         ✔️   |       ❌️     | 2 different instructions |


# Test & score

## Test & score

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Evaluate model | ✔️    |       ✔️      |          |


# Bundle

## Losses

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Mean Squared Error |✔️ |      ✔️       |          |
| BinaryCrossEntropy |✔️ |      ✔️       |          |
| CategoricalCrossEntropy |✔️ | ✔️       |          |
| Soft Cross-Entropy |✔️ |      ✔️       |          |
| Dice               |❌️|      ✔️       |          |


## Metrics

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Mean Squared Error |❌️|       ❌️     |          |
| Categorical Accuracy |❌️|     ❌️     |          |
| Mean Absolute Error |❌️|      ❌️     |          |
| Mean Relative Error |❌️|      ❌️     |          |


## Regularizers

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| L1      |      ✔️      |       ❌️     |          | 
| L2      |      ✔️      |       ❌️     |          |
| L1L2    |      ✔️      |       ❌️     |          |


## Initializers

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| GlorotNormal |    ✔️   |       ❌️     |          |
| GlorotUniform |   ✔️   |       ❌️     |          |
| RandomNormal |    ✔️   |       ❌️     |          |
| RandomUniform |   ✔️   |       ❌️     |          |
| Constant |        ✔️   |       ❌️     |          |


## Optimizers

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Adadelta |     ✔️      |       ❌️     |          |
| Adam    |      ✔️      |       ❌️     |          |
| Adagrad |      ✔️      |       ❌️     |          |
| Adamax  |      ✔️      |       ❌️     |          |
| Nadam   |      ✔️      |       ❌️     |          |
| RMSProp |      ✔️      |       ❌️     |          |
| SGD (Stochastic Gradient Descent) |✔️| ❌️|          |


# Computing services

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| CPU     |      ✔️      |       ✔️      | check use of English |
| GPU     |      ✔️      |       ✔️      | many examples - maybe improve leaving just the one with the most params and marking them as optional? |
| FPGA    |      ❌️     |       ❌️     | note: not yet implemented |
| COMPSS  |      ✔️      |       ❌️     | note: not yet implemented |


# Datasets

## Classification

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| MNIST   |      ✔️      |       ✔️      |          |
| CIFAR   |      ✔️      |       ✔️      |          |

## Segmentation

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| DRIVE   |      ✔️      |       ❌️     |          |


# Tensor

## Creation Routines

### Constructors


| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| Tensor  |      ✔️      |       ✔️      |          |

### Constructors & Initializers


| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| empty   |      ✔️      |       ✔️      |          |
| empty_like|    ✔️      |       ✔️      |          |
| zeros   |      ✔️      |       ✔️      |          |
| zeros_like|    ✔️      |       ✔️      |          |
| ones    |      ✔️      |       ✔️      |          |
| ones_like|     ✔️      |       ✔️      |          |
| full    |      ✔️      |       ✔️      |          |
| full_like|     ✔️      |       ✔️      |          |
| eye     |      ✔️      |       ✔️      |          |
| identity|      ✔️      |       ✔️      |          |

### Constructors from existing data

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| clone   |      ✔️      |       ✔️      |          |
| reallocate|    ✔️      |       ✔️      |          |
| copy    |      ✔️      |       ✔️      |          |

### Constructors from numerical ranges

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| arange  |      ✔️      |       ✔️      |          |
| range   |      ✔️      |       ✔️      |          |
| linspace|      ✔️      |       ✔️      |          |
| logspace|      ✔️      |       ✔️      |          |
| geomspace|     ✔️      |       ✔️      |          |

### Constructors from random generators

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| randu   |      ✔️      |       ✔️      |          |
| randn   |      ✔️      |       ✔️      |          |

### Constructors of matrices

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| diag    |      ✔️      |       ✔️      |          |


## Manipulation

### Devices and information

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| toCPU   |      ❌️     |       ✔️      | needs comments in the .h |
| toGPU   |      ❌️     |       ✔️      | needs comments in the .h |
| isCPU   |      ❌️     |       ✔️      | needs better comments in the .h |
| isGPU   |      ❌️     |       ✔️      | needs better comments in the .h |
| isFPG   |      ❌️     |       ✔️      | needs better comments in the .h |
| getDeviceName| ❌️     |       ✔️      | needs better comments in the .h |
| info    |      ✔️      |       ✔️      |          |
| print   |      ✔️      |       ✔️      |          |
| isSquared|     ✔️      |       ✔️      |          |

### Changing array shape

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| reshape |      ✔️      |       ✔️      |          |
| flatten |      ✔️      |       ✔️      |          |

### Transpose-like operations

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| permute |      ✔️      |       ✔️      |          |
| moveaxis|      ✔️      |       ✔️      |          |
| swapaxis|      ✔️      |       ✔️      |          |

### Changing number of dimensions

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| squeeze |      ✔️      |       ✔️      |          |
| unsqueeze|     ✔️      |       ✔️      |          |

### Joining arrays

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| concat  |      ❌️     |      ✔️      | needs comments in the .h |

### Value operations

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| fill    |      ✔️      |       ✔️      |          |
| fill_rand_uniform_| ✔️ |       ✔️      |          |
| fill_rand_signed_uniform_|✔️|  ✔️      |          |
| fill_rand_normal_| ✔️  |       ✔️      |          |
| fill_rand_binary _| ✔️ |       ✔️      |          |


## Image operations

* Note about practical examples

### Transformations

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| shift   |      ✔️      |       ✔️      |          |
| rotate  |      ✔️      |       ✔️      |          |
| scale   |      ✔️      |       ✔️      |          |
| flip    |      ✔️      |       ✔️      |          |
| crop    |      ✔️      |       ✔️      |          |
| crop_scale|    ✔️      |       ✔️      |          |
| cutout  |      ✔️      |       ✔️      |          |

### Data augmentations

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| shift_random|  ✔️      |       ✔️      |          |
| rotate_random| ✔️      |       ✔️      |          |
| scale_random|  ✔️      |       ✔️      |          |
| flip_random|   ✔️      |       ✔️      |          |
| crop_random|   ✔️      |       ✔️      |          |
| crop_scale_random| ✔️  |       ✔️      |          |
| cutout_random| ✔️      |       ✔️      |          |


## Indexing & Sorting

### Indexing

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| nonzero |      ✔️      |       ✔️      |          |
| where   |      ✔️      |       ✔️      |          |
| select  |      ✔️      |       ✔️      |          |
| set_select|    ❌️     |       ✔️      | explanation of parameters not complete in the .h |

### Sorting

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| sort    |      ✔️      |       ✔️      |          |
| argsort |      ✔️      |       ✔️      |          |


## Input/Output Operations

* Note about practical examples

### Input

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| load    |      ✔️      |       ✔️      |          |

### Output

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| save    |      ✔️      |       ✔️      |          |


## Linear algebra

### Matrix and vector operations

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| interpolate|   ✔️      |       ✔️      |          |
| trace   |      ✔️      |       ✔️      |          |


## Logic functions

* Note about practical examples

### Truth value testing

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| all     |      ✔️      |       ✔️      |          |
| any     |      ✔️      |       ✔️      |          |

### Array contents

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| isfinite|      ✔️      |       ✔️      |          |
| isinf   |      ✔️      |       ✔️      |          |
| isnan   |      ✔️      |       ✔️      |          |
| isneginf|      ✔️      |       ✔️      |          |
| isposinf|      ✔️      |       ✔️      |          |

### Logical operations

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| logical_and|   ✔️      |       ✔️      |          |
| logical_or|    ✔️      |       ✔️      |          |
| logical_not|   ✔️      |       ✔️      |          |
| logical_xor|   ✔️      |       ✔️      |          |

### Comparison

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| greater |      ✔️      |       ✔️      |          |
| greater_equal| ✔️      |       ✔️      |          |
| less    |      ✔️      |       ✔️      |          |
| less_equal|    ✔️      |       ✔️      |          |
| equal   |      ✔️      |       ✔️      |          |
| not_equal|     ✔️      |       ✔️      |          |

### Binary Operations

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| allclose|      ✔️      |       ✔️      |          |
| isclose |      ✔️      |       ✔️      |          |
| greater |      ✔️      |       ✔️      |          |
| greater_equal| ✔️      |       ✔️      |          |
| less    |      ✔️      |       ✔️      |          |
| less_equal|    ✔️      |       ✔️      |          |
| equal   |      ✔️      |       ✔️      |          |
| not_equal|     ✔️      |       ✔️      |          |


## Mathematical functions

### Point-wise

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| abs     |      ✔️      |       ✔️      |          |
| acos    |      ✔️      |       ✔️      |          |
| add     |      ✔️      |       ✔️      |          |
| asin    |      ✔️      |       ✔️      |          |
| atan    |      ✔️      |       ✔️      |          |
| ceil    |      ✔️      |       ✔️      |          |
| clamp   |      ✔️      |       ✔️      |          |
| clampmax|      ✔️      |       ✔️      |          |
| clampmin|      ✔️      |       ✔️      |          |
| cos     |      ✔️      |       ✔️      |          |
| cosh    |      ✔️      |       ✔️      |          |
| div     |      ✔️      |       ✔️      |          |
| exp     |      ✔️      |       ✔️      |          |
| floor   |      ✔️      |       ✔️      |          |
| inv     |      ✔️      |       ✔️      |          |
| log     |      ✔️      |       ✔️      |          |
| log2    |      ✔️      |       ✔️      |          |
| log10   |      ✔️      |       ✔️      |          |
| logn    |      ✔️      |       ✔️      |          |
| maximum |      ✔️      |       ✔️      |          |
| minimum |      ✔️      |       ✔️      |          |
| mod     |      ✔️      |       ✔️      |          |
| mult    |      ✔️      |       ✔️      |          |
| neg     |      ✔️      |       ✔️      |          |
| normalize|     ✔️      |       ✔️      |          |
| pow     |      ✔️      |       ✔️      |          |
| powb    |      ✔️      |       ✔️      |          |
| reciprocal|    ✔️      |       ✔️      |          |
| remainder|     ✔️      |       ✔️      |          |
| round   |      ✔️      |       ✔️      |          |
| rsqrt   |      ✔️      |       ✔️      |          |
| sigmoid |      ✔️      |       ✔️      |          |
| sign    |      ✔️      |       ✔️      |          |
| sin     |      ✔️      |       ✔️      |          |
| sinh    |      ✔️      |       ✔️      |          |
| sqr     |      ✔️      |       ✔️      |          |
| sqrt    |      ✔️      |       ✔️      |          |
| sub     |      ✔️      |       ✔️      |          |
| tan     |      ✔️      |       ✔️      |          |
| tanh    |      ✔️      |       ✔️      |          |
| trunc   |      ✔️      |       ✔️      |          |

### Element-wise

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| add     |      ✔️      |       ✔️      |          |
| div     |      ✔️      |       ✔️      |          |
| maximum |      ✔️      |       ✔️      |          |
| minimum |      ✔️      |       ✔️      |          |
| mult    |      ✔️      |       ✔️      |          |
| sub     |      ✔️      |       ✔️      |          |

### Reductions

| Section | Explanation | Instructions | Comments |
| :------ | :---------: | :----------: | :------- |
| argmax  |      ✔️      |       ✔️      |          |
| argmin  |      ✔️      |       ✔️      |          |
| max     |      ✔️      |       ✔️      |          |
| mean    |      ✔️      |       ✔️      |          |
| median  |      ✔️      |       ✔️      |          |
| min     |      ✔️      |       ✔️      |          |
| mode    |      ✔️      |       ✔️      |          |
| norm    |      ✔️      |       ✔️      |          |
| prod    |      ✔️      |       ✔️      |          |
| std     |      ✔️      |       ✔️      |          |
| sum     |      ✔️      |       ✔️      |          |
| sum_abs |      ✔️      |       ✔️      |          |
| var     |      ✔️      |       ✔️      |          |
