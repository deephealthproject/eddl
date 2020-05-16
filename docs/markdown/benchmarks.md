# Benchmarks

Benchmarks for development. 

> Disclaimer: This is not a serious benchmark. The results in this sections are intendeed for the developers of this
> library so that can easily check if their results are consistent with previous versions.


## Desktop - Ubuntu 18.04 - AMD Ryzen 7 2700X Eight-Core Processor - 4.3Ghz (16 cores) - 16GB RAM - GeForce GTX 1070 (8GB)

### MNIST MLP (`1_mnist_mlp.cpp`)


#### CPU only

**Default flags:**

```
Setup
-------
TARGET: CPU
CORES: 16
EPOCHS: 1
C++ flags (release): -O3

Training/Evaluation:
--------------------


Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
```


**Optimization flags:**

```
Setup
-------
TARGET: CPU
CORES: 16
EPOCHS: 1
C++ flags (release): -O3 -Ofast -msse -mfpmath=sse -ffast-math

Training/Evaluation:
--------------------


Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
```

```
Setup
-------
TARGET: CPU
CORES: 16
EPOCHS: 1
C++ flags (release): -O3

Training/Evaluation:
--------------------


Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
```

```
Setup
-------
TARGET: CPU
CORES: 16
EPOCHS: 1
C++ flags (release): -O3 -Ofast -msse -mfpmath=sse -ffast-math

Training/Evaluation:
--------------------


Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
```


### CIFAR10 CONV (`1_cifar_conv.cpp`)

#### CPU only

**Default flags:**

```
Setup
-------
TARGET: CPU
CORES: 8
EPOCHS: 1
C++ flags (release): -O3

Training/Evaluation:
--------------------


Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
```


## MacBook Pro 2013 - macOS Catalina (version 10.15 - Quad-Core Intel Core i7 - 2.3Ghz (8 cores) - 8GB RAM - No GPU

### MNIST MLP (`1_mnist_mlp.cpp`)


#### CPU only

**Default flags:**

```
Setup
-------
TARGET: CPU
CORES: 8
EPOCHS: 1
C++ flags (release): -O3

Training/Evaluation:
--------------------
Batch 600 softmax4 loss[soft_cross_entropy]=0.085 metric[categorical_accuracy]=0.842  -- 0.068 secs/batch
40.986 secs/epoch
Evaluate with batch size 100
Batch 100 softmax4 loss[soft_cross_entropy]=0.034 metric[categorical_accuracy]=0.945  -- 

Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
93492 salvacarr  17   0 4616M  271M ? 714.  3.3  1:42.05 ./bin/mnist_mlp
```


**Optimization flags:**

```
Setup
-------
TARGET: CPU
CORES: 8
EPOCHS: 1
C++ flags (release): -O3 -Ofast -msse -mfpmath=sse -ffast-math

Training/Evaluation:
--------------------
Batch 600 softmax4 loss[soft_cross_entropy]=0.082 metric[categorical_accuracy]=0.843  -- 0.078 secs/batch
46.648 secs/epoch
Evaluate with batch size 100
Batch 100 softmax4 loss[soft_cross_entropy]=0.029 metric[categorical_accuracy]=0.951  -- 

Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
98456 salvacarr  17   0 4616M  271M ? 616.  3.3  2:32.09 ./bin/mnist_mlp
```


### CIFAR10 CONV (`1_cifar_conv.cpp`)

#### CPU only

**Default flags:**

```
Setup
-------
TARGET: CPU
CORES: 8
EPOCHS: 1
C++ flags (release): -O3

Training/Evaluation:
--------------------
Batch 500 softmax5 loss[soft_cross_entropy]=0.293 metric[categorical_accuracy]=0.254  -- 1.024 secs/batch
512.078 secs/epoch
Evaluate test:
Evaluate with batch size 100
Batch 100 softmax5 loss[soft_cross_entropy]=0.255 metric[categorical_accuracy]=0.373  -- 

Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
94687 salvacarr  17   0 5225M  814M ? 721.  9.9 33:38.78 ./bin/cifar_conv
```