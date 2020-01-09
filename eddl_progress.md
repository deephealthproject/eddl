### Development Status

| Image | Meaning |
| ------------- |------|
| ✅ | Done |
| 🔵 | In progress |
| ❌ | Todo |


# Layers
---

## Core layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Dense | ✅ | ✅ | Just your regular densely-connected NN layer. |
| Activation | ✅ | ✅ | Applies an activation function to an output: ReLu, LReLu, Softmax, Sigmoid, Tanh |
| Dropout | ✅ | ✅ | Applies Dropout to the input. |
| Flatten | 🔵 | 🔵 | Flattens the input. Does not affect the batch size. (Wrapper for Reshape) |
| Input | ✅ | ✅ | Used to instantiate a EDDL tensor. |
| Reshape | ✅ | ✅ | Reshapes an output to a certain shape. |
| Permute | ✅ | ✅ | Permutes the dimensions of the input according to a given pattern. |
| Embedding | ❌ | ❌ | |
| Transpose | ✅ | ✅ | |


## Convolutional layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Conv2D | ✅ | ✅ | Dilated, Groups... |
| Conv2DT | ❌ | ❌ | |
| UpSampling | ✅ | ✅ | |


## Data augmentation layers

Image transformations with random values define in a range

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Crop (random) | ✅ | ✅ | |
| Crop & Scale (random) | ✅ | ✅ | |
| Cutout (random) | ✅ | ✅ | |
| Flip (random) | ✅ | ✅ | |
| Rotate (random) |  ✅ | ✅ | |
| Scale (random) | ✅ | ✅ | |
| Shift (random) | ✅ | ✅ | |


## Data transformation

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Crop | ✅ | ✅ | |
| Crop & Scale | ✅ | ✅ | |
| Cutout | ✅ | ✅ | |
| Flip | ✅ | ✅ | |
| Rotate|  ✅ | ✅ | |
| Scale | ✅ | ✅ | |
| Shift | ✅ | ✅ | |


## Merge layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Add | ✅ | ✅ | Layer that adds a list of inputs. |
| Substract | ✅ | ✅ | Layer that subtracts two inputs. |
| MatMul | ✅ | ✅ | Layer that multiplies (element-wise) a list of inputs. |
| Average | ✅ | ✅ | Layer that averages a list of inputs. |
| Concat | ✅ | ✅ | Layer that concatenates a list of inputs. |
| Max | ✅ | ✅ | Layer that computes the maximum (element-wise) a list of inputs. |
| Min | ✅ | ✅ | Layer that computes the minimum (element-wise) a list of inputs. |


## Normalization

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| BatchNorm | ✅ | ✅ | Batch normalization layer (Ioffe and Szegedy, 2014).  |


## Noise layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Gaussian | ✅ | ✅ |
| Uniform | ❌| ❌ | still test properly


## Pooling layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| AvgPool | ❌ | ❌ |
| GlobalMaxPool | ❌ | ❌ |
| MaxPool | ✅ | ✅ |


## Operators layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Abs |  ✅ | ✅ | |
| Diff | ✅ | ✅ | |
| Div | ✅ | ✅ | |
| Exp | ✅ | ✅ | |
| Log | ✅ | ✅ | |
| Log2 |  ✅ | ✅ | |
| Log10 | ✅ | ✅ | |
| Mult | ✅ | ✅| |
| Pow |  ✅ | ✅ | |
| Select |  ✅ | ✅ | |
| Sqrt |  ✅ | ✅ | |
| Sum | ✅ | ✅ | |


## Reduction layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Max | ✅| ✅ | |
| Mean | ✅| ✅ | |
| Min | ✅| ✅ | |
| Sum | ✅| ✅ | |
| Var | ✅| ✅ | |


## Recurrent layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| LSTM | ❌ | ❌ |
| RNN | ❌ | ❌ |


# Initializers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Constant |  ✅ | ✅ | |
| GlorotNormal |  ✅ | ✅ | |
| GlorotUniform |  ✅ | ✅ | |
| Identity | ❌ | ❌ | |
| Orthogonal | ❌ | ❌ | |
| RandomNormal |  ✅ | ✅ | |
| RandomUniform |  ✅ | ✅ | |


# Constraints

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| MaxNorm |  ❌ | ❌ | |
| MinMaxNorm |  ❌ | ❌ | |
| NonNeg |  ❌ | ❌ | |
| UnitNorm |  ❌ | ❌ | |


# Loss functions

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| CrossEntropy | ✅ | ✅ | |
| MSE | ✅ | ✅ | |
| Min | ✅ | ✅ | |
| SoftCE | ✅ | ✅ | |


# Metric functions

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| CategoricalAcc | ✅ | ✅ | |
| MSA | ✅ | ✅ | |
| MRE | ✅ | ✅ | |
| MSE | ✅ | ✅ | |
| MSum | ✅ | ✅ | |


# Optimizers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Adadelta |✅ | ✅ | |
| Adagrad | ✅ | ✅ | |
| Adam | ✅ | ✅ |
| Adaax | ✅ | ✅ | |
| Nadam | ✅ | ✅ | |
| RMSProp |✅ | ✅ |
| SGD | ✅ | ✅ |
