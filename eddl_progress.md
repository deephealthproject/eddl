### Development Status

| Image | Meaning |
| ------------- |------|
| ✅ | Done |
| 🔵 | In progress |
| ❌ | Todo |


---
# Layers
---

## Core layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Tensor | ✅ | ✅ | |
| Dense | ✅ | ✅ | |
| Activation | 🔵 | 🔵 | Sigmoid, LReLu ...
| BatchNorm | ✅ | ✅ |
| Embedding | ❌ | ❌ |
| Input | ✅ | ✅ | |
| Reshape | ✅ | ✅ | |
| Transpose | ❌ | ❌ |
| Drop | ✅ | ✅ |


## Convolutional layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Conv2D | ✅ | ✅ | Dilated, Groups...
| Conv2DT | ❌ | ❌ |
| Upsampling | ✅ | ✅ |


## Pooling layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| AvgPool | ❌ | ❌ |
| GlobalMaxPool | ❌ | ❌ |
| MaxPool | ✅ | ✅ |


## Merge layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Add | ✅ | ✅ |
| Average | ❌ | ❌ |
| Concat | ✅ | ✅ |
| MatMul | ❌ | ❌ |
| Max | ❌ | ❌ |
| Merge | ❌ | ❌ |
| Min | ❌ | ❌ |
| Substract | ❌ | ❌ |


## Noise layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Gaussian | ✅ | ✅ |
| Uniform | ❌| ❌ | still test properly


## Operators layers

> **Note:** Do not confuse with raw-tensor operations

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Abs | ✅ | ❌ |
| Diff | ✅ | ✅ |
| Div | ✅ | ✅ |
| Exp | ✅ | ✅ |
| Log | ✅ | ✅ |
| Log10 | ✅ | ❌|
| Log2 | ✅ | ❌ |
| Mult | ✅ | ✅|
| Pow | ❌ | ❌ |
| Sqrt | ❌ | ❌ |
| Sum | ✅ | ✅ |


## Reduction layers

> **Note:** Do not confuse with raw-tensor reductions

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Mean | ✅| ✅ |
| Var | ✅| ✅ |
| Sum | ✅| ✅ |
| Max | ✅| ✅ |
| Min | ✅| ✅ |


## Recurrent layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| LSTM | ❌ | ❌ |
| RNN | ❌ | ❌ |


---
# Initializers
---

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Constant | ❌ | ❌ |
| GlorotNormal | ❌ | ❌ |
| GlorotUniform | ❌ | ❌ |
| Identity | ❌ | ❌ |
| Orthogonal | ❌ | ❌ |
| RandomNormal | ❌ | ❌ |
| RandomUniform | ❌ | ❌ |


---
# Loss functions
---

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| CrossEntropy | ✅ | ✅ |
| MSE | ✅ | ✅ |
| SoftCE | ✅ | ✅ |


---
# Metric functions
---

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| CategoricalAcc | ✅ | ✅ |
| MSE | ✅ | ✅ |


---
# Optimizers
---

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Adadelta | ❌ | ❌ |
| Adagrad | ❌ | ❌ |
| Adam | ❌ | ❌ |
| Adamax | ❌ | ❌ |
| Nadam | ❌ | ❌ |
| RMSProp | ❌ | ❌ |
| SGD | ✅ | ✅ |
