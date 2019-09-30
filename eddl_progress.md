# Development Status
---

| Image | Meaning |
| ------------- |------| 
| ✅ | Done | 
| 🔵 | In progress | 
| ❌ | Todo | 


## Layers
---

### Core layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Tensor | ✅ | ✅ | |
| Dense | ✅ | ✅ | |
| Activation | 🔵 | 🔵 | Sigmoid, LReLu ...
| BatchNorm | ❌ | ❌ |
| Embedding | ❌ | ❌ |
| Input | ✅ | ✅ | |
| Reshape | ✅ | ✅ | |
| Transpose | ❌ | ❌ |
| Drop | ✅ | ✅ |


### Convolutional layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Conv2D | 🔵 | 🔵 | Dilated, Groups...
| Conv2DT | ❌ | ❌ |
| Upsampling | ❌ | ❌ |
| AvgPool | ❌ | ❌ |
| GlobalMaxPool | ❌ | ❌ |
| MaxPool | ✅ | ✅ |


### Pooling layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| AvgPool | ❌ | ❌ |
| GlobalMaxPool | ❌ | ❌ |
| MaxPool | ✅ | ✅ |


### Merge layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Add | ✅ | ✅ |
| Average | ❌ | ❌ |
| Concat | ❌ | ❌ |
| MatMul | ❌ | ❌ |
| Max | ❌ | ❌ |
| Merge | ❌ | ❌ |
| Min | ❌ | ❌ |
| Substract | ❌ | ❌ |


### Noise layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Gaussian | ✅ | ✅ |
| Uniform | ❌| ❌ | still test properly


### Operators layers

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


### Reduction layers

> **Note:** Do not confuse with raw-tensor reductions

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| Mean | ✅| ❌ | still test properly
| Var | ✅| ❌ | still test properly
| Sum | ❌| ❌ | still test properly
| Max | ❌| ❌ | still test properly
| Min | ❌| ❌ | still test properly


### Recurrent layers

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| LSTM | ❌ | ❌ |
| RNN | ❌ | ❌ |


## Initializers
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


## Loss functions
---

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| CrossEntropy | ✅ | ✅ |
| MSE | ✅ | ✅ |
| SoftCE | ✅ | ✅ |


## Metric functions
---

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| CategoricalAcc | ✅ | ✅ |
| MSE | ✅ | ✅ |


## Optimizers
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


## Raw-Tensor operations
---

Numpy-like operations over a raw-tensor object

### Creation ops


| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| zeros | ✅ | ✅ |
| ones | ✅ | ✅ |
| arange | ✅ | ✅ |
| range | ✅ | ✅ |
| linspace | ✅ | ✅ |
| logspace | ✅ | ✅ |
| eye | ✅ | ✅ |
| full | ✅ | ✅ |


### Indexing, Slicing, Joining, Mutating Ops

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| cat | ❌ | ❌ |
| chunk | ❌ | ❌ |
| gather | ❌ | ❌ |
| nonzero | ❌ | ❌ |
| reshape | ✅ | ✅ |
| split | ❌ | ❌ |
| squeeze | ❌ | ❌ |
| stack | ❌ | ❌ |
| transpose | ✅ | ✅ |
| unsqueeze | ❌ | ❌ |
| where | ❌ | ❌ |
| get | ❌ | ❌ |
| set | ❌ | ❌ |


### Generators

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| bernoulli | ❌ | ❌ |
| multinomial | ❌ | ❌ |
| uniform | ✅ | ✅ |
| signed-uniform | ✅ | ✅ |
| rand normal | ✅ | ✅ |
| rand binary | ✅ | ✅ |


### Serialization

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| save | ✅ | ✅ |
| load | ✅ | ✅ |


### Math operations

#### Pointwise Ops

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| abs | ✅ | ❌ |
| acos | ✅ | ❌ |
| add | ✅ | ✅ |
| asin | ✅ | ❌ |
| atan | ✅ | ❌ |
| ceil | ✅ | ❌ |
| clamp | ✅ | ❌ |
| clampmax | ✅ | ❌ |
| clampmin | ✅ | ❌ |
| cos | ✅ | ❌ |
| cosh | ✅ | ❌ |
| div | ✅ | ✅ |
| exp | ✅ | ✅ |
| floor | ✅ | ❌ |
| log | ✅ | ✅ |
| log2 | ✅ | ✅ |
| log10 | ✅ | ✅ |
| logn | ✅ | ✅ |
| max* | ✅ | ❌ | Not reduced
| mean* | ❌ | ❌ | Not reduced
| median* | ❌ | ❌ | Not reduced
| min* | ✅ | ❌ | Not reduced
| mod | ✅ | ❌ |
| mode* | ✅ | ❌ | Not reduced
| mult | ✅ | ✅ |
| neg | ✅ | ❌ |
| normalize* | ✅ | ❌ | Not reduced
| pow | ✅ | ✅ |
| reciprocal | ✅ | ❌ |
| remainder | ✅ | ❌ |
| round | ✅ | ❌ |
| rsqrt | ✅ | ❌ |
| sigmoid | ✅ | ❌ |
| sign | ✅ | ❌ |
| sin | ✅ | ❌ |
| sinh | ✅ | ❌ |
| sqr | ✅ | ✅ |
| sqrt | ✅ | ✅ |
| std* | ❌ | ❌ | Not reduced
| sub | ✅ | ❌ |
| sum* | ✅ | ❌ | Not reduced by default
| tan | ✅ | ❌ |
| tanh | ✅ | ❌ |
| trunc | ✅ | ❌ |
| var* | ❌ | ❌ | Not reduced


#### Reduction ops

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| argmax | ❌ | ❌ |
| argmin | ❌ | ❌ |
| cumprod | ❌ | ❌ |
| cumsum | ❌ | ❌ |
| max | ❌ | ❌ |
| min | ❌ | ❌ |
| mean | ❌ | ❌ |
| median | ❌ | ❌ |
| mode | ❌ | ❌ |
| norm | ❌ | ❌ |
| prod | ❌ | ❌ |
| std | ❌ | ❌ |
| sum | ❌ | ❌ |
| unique | ❌ | ❌ |
| var | ❌ | ❌ |


#### Comparison ops

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| allclose | ❌ | ❌ |
| argsort | ❌ | ❌ |
| eq | ✅ | ❌ |
| ge | ❌ | ❌ |
| gt | ❌ | ❌ |
| isfinite | ❌ | ❌ |
| isinf | ❌ | ❌ |
| isnan | ❌ | ❌ |
| le | ❌ | ❌ |
| lt | ❌ | ❌ |
| ne | ❌ | ❌ |
| sort | ❌ | ❌ |
| topk | ❌ | ❌ |


#### Other ops

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| cross | ❌ | ❌ |
| diag | ❌ | ❌ |
| einsum | ❌ | ❌ |
| flatten | ❌ | ❌ |
| flip | ❌ | ❌ |
| trace | ❌ | ❌ |
| dot | ❌ | ❌ |