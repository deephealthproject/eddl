# Benchmarks

Benchmarks for development. 

> Disclaimer: This is not a serious benchmark. The results in this sections are intendeed for the developers of this
> library so that can easily check if their results are consistent with previous versions.


## Desktop - Ubuntu 18.04 - AMD Ryzen 7 2700X Eight-Core Processor - 4.3Ghz (16 cores) - 16GB RAM - GeForce GTX 1070 (8GB)

### MNIST MLP (`1_mnist_mlp.cpp`)

#### GPU

**Default flags:**

```
Setup
-------
VERSION: v0.7
TARGET: CPU
CORES: 16
EPOCHS: 1
C++ flags (release): -O3

Training/Evaluation:
--------------------
1 epochs of 600 batches of size 100
Epoch 1
Batch 600 softmax4 ( loss[soft_cross_entropy]=0.082 metric[categorical_accuracy]=0.857 ) -- 0.004 secs/batch
2.606 secs/epoch
Evaluate with batch size 100
Batch 100 softmax4 ( loss[soft_cross_entropy]=0.031 metric[categorical_accuracy]=0.949 ) -- 

Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
22164 salvaca+  20   0 10,463g 585428 190712 S  1590  3,6   1:46.01 mnist_mlp                                                                                                                                          

GPU Memory:
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.82       Driver Version: 440.82       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1070    Off  | 00000000:09:00.0  On |                  N/A |
| 54%   71C    P2    76W / 190W |   1188MiB /  8118MiB |     71%      Default |
+-------------------------------+----------------------+----------------------+

```

#### CPU only

**Default flags:**

```
Setup
-------
VERSION: v0.7
TARGET: CPU
CORES: 16
EPOCHS: 1
C++ flags (release): -O3

Training/Evaluation:
--------------------
1 epochs of 600 batches of size 100
Epoch 1
Batch 600 softmax4 ( loss[soft_cross_entropy]=0.083 metric[categorical_accuracy]=0.848 ) -- 0.245 secs/batch
147.224 secs/epoch
Evaluate with batch size 100
Batch 100 softmax4 ( loss[soft_cross_entropy]=0.046 metric[categorical_accuracy]=0.927 ) -- 

Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
15057 salvaca+  20   0 2643176 290468  17128 S  1569  1,8   1:12.03 mnist_mlp                                                                                                                                          

```


**Optimization flags:**

- Test 1: `-Ofast -msse -mfpmath=sse -ffast-math`

```
Setup
-------
VERSION: v0.7
TARGET: CPU
CORES: 16
EPOCHS: 1
C++ flags (release): -O3 -Ofast -msse -mfpmath=sse -ffast-math
=========================> ERROR!

Training/Evaluation:
--------------------
1 epochs of 600 batches of size 100
Epoch 1
Batch 600 softmax4 ( loss[soft_cross_entropy]=-nan metric[categorical_accuracy]=0.128 ) -- 0.240 secs/batch
144.023 secs/epoch
Evaluate with batch size 100
Batch 100 softmax4 ( loss[soft_cross_entropy]=nan metric[categorical_accuracy]=0.101 ) -- 

Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
18701 salvaca+  20   0 2643196 290872  17536 S  1581  1,8   2:56.98 mnist_mlp                                                                                                                                          
```


- Test 2: `-mtune=native`

```
Setup
-------
VERSION: v0.7
TARGET: CPU
CORES: 16
EPOCHS: 1
C++ flags (release): -mtune=native
=====================> There are problems with this flag

Training/Evaluation:
--------------------
1 epochs of 600 batches of size 100
Epoch 1
Batch 600 softmax4 ( loss[soft_cross_entropy]=-nan metric[categorical_accuracy]=0.102 ) -- 0.256 secs/batch
153.884 secs/epoch
Evaluate with batch size 100
Batch 100 softmax4 ( loss[soft_cross_entropy]=-nan metric[categorical_accuracy]=0.098 ) -- 


Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
 3690 salvaca+  20   0 2643176 290276  16916 S  1531  1,8   4:06.69 mnist_mlp   
```

- Test 3: `-march=native`
```
Setup
-------
VERSION: v0.7
TARGET: CPU
CORES: 16
EPOCHS: 1
C++ flags (release): -mtune=native
=====================> 


1 epochs of 500 batches of size 100
Epoch 1

Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
```

##### v0.8 improvements


| HPC | Softmax | Loss      | Time  | Comments      |
|-----|---------|-----------|-------|---------------|
| OFF | Old     | (old) CE  | 0.043 | Epoch1: 0.247 |
| OFF | Old     | (new) CE  | 0.041 | Epoch1: 0.099 |
| OFF | Old     | (old) SCE | 0.040 | Epoch1: 0.347 |
| OFF | Old     | (new) SCE | 0.037 | Epoch1: 0.901 |
| OFF | New     | (new) CE  | 0.037 | Epoch1: 0.900 |
| OFF | New     | (new) SCE | 0.036 | Epoch1: 0.903 |
| ON  | New     | (new) SCE | 0.024 | Epoch1: 0.901 |

> Desktop - Ubuntu 18.04 - AMD Ryzen 7 2700X Eight-Core Processor - 4.3Ghz (16 cores) - 16GB RAM - GeForce GTX 1070 (8GB)
> MNIST MLP (`1_mnist_mlp.cpp`)

  
### CIFAR10 CONV (`1_cifar_conv.cpp`)

#### GPU

**Default flags:**

```
Setup
-------
VERSION: v0.7
TARGET: CPU
CORES: 16
EPOCHS: 1
C++ flags (release): -O3

Training/Evaluation:
--------------------
1 epochs of 500 batches of size 100
Epoch 1
Batch 500 softmax5 ( loss[soft_cross_entropy]=0.274 metric[categorical_accuracy]=0.318 ) -- 0.014 secs/batch
7.099 secs/epoch
Evaluate test:
Evaluate with batch size 100
Batch 100 softmax5 ( loss[soft_cross_entropy]=0.230 metric[categorical_accuracy]=0.454 ) -- 

Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
12366 salvaca+  20   0 11,137g 1,011g 191004 R  1562  6,5   0:28.25 cifar_conv                                                                                                                                                                                                                                                                               

GPU Memory:
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.82       Driver Version: 440.82       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1070    Off  | 00000000:09:00.0  On |                  N/A |
| 58%   77C    P2    88W / 190W |   1369MiB /  8118MiB |     94%      Default |
+-------------------------------+----------------------+----------------------+


```


#### CPU only

**Default flags:**

```
Setup
-------
VERSION: v0.7
TARGET: CPU
CORES: 16
EPOCHS: 1
C++ flags (release): -O3

Training/Evaluation:
--------------------
1 epochs of 500 batches of size 100
Epoch 1
Batch 500 softmax5 ( loss[soft_cross_entropy]=0.325 metric[categorical_accuracy]=0.101 ) -- 0.281 secs/batch
140.446 secs/epoch
Evaluate test:
Evaluate with batch size 100
Batch 100 softmax5 ( loss[soft_cross_entropy]=0.325 metric[categorical_accuracy]=0.100 ) -- 

Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
13661 salvaca+  20   0 3258216 919352  16960 S  1588  5,6   1:37.88 cifar_conv                                                                                                                                         

```

**Optimization flags:**

- Test 1: `-Ofast -msse -mfpmath=sse -ffast-math`

```
Setup
-------
VERSION: v0.7
TARGET: CPU
CORES: 16
EPOCHS: 1
C++ flags (release): -O3 -Ofast -msse -mfpmath=sse -ffast-math

Training/Evaluation:
--------------------
1 epochs of 500 batches of size 100
Epoch 1
Batch 500 softmax5 ( loss[soft_cross_entropy]=0.297 metric[categorical_accuracy]=0.239 ) -- 0.302 secs/batch
151.189 secs/epoch
Evaluate test:
Evaluate with batch size 100
Batch 100 softmax5 ( loss[soft_cross_entropy]=0.260 metric[categorical_accuracy]=0.369 ) -- 

Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
20308 salvaca+  20   0 3258236 919972  17584 S  1569  5,6  32:09.53 cifar_conv                                                                                                                                         
```

- Test 2: `-mtune=native`

```
Setup
-------
VERSION: v0.7
TARGET: CPU
CORES: 16
EPOCHS: 1
C++ flags (release): -mtune=native
=====================> There are problems with this flag

Training/Evaluation:
--------------------
1 epochs of 500 batches of size 100
Epoch 1
Batch 500 softmax5 ( loss[soft_cross_entropy]=0.296 metric[categorical_accuracy]=0.242 ) -- 0.289 secs/batch
144.295 secs/epoch
Evaluate test:
Evaluate with batch size 100
Batch 100 softmax5 ( loss[soft_cross_entropy]=0.259 metric[categorical_accuracy]=0.373 ) -- 


Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
 5471 salvaca+  20   0 3258216 919248  16848 S  1532  5,6  10:41.81 cifar_conv  

```

- Test 3: `-march=native`
```
Setup
-------
VERSION: v0.7
TARGET: CPU
CORES: 16
EPOCHS: 1
C++ flags (release): -mtune=native
=====================> 


1 epochs of 500 batches of size 100
Epoch 1

Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
```


## Desktop - MacOS Catalina - AMD Ryzen 7 2700X Eight-Core Processor - 3,4 GHz Intel Core i5 de 4 núcleos - 8 GB 2400 MHz DDR4

### MNIST MLP (`1_mnist_mlp.cpp`)

#### CPU


**Optimization flags:**

```
Setup
-------
TARGET: CPU
CORES: 4
EPOCHS: 1
C++ flags:
-- C++ compiler: AppleClang (/Library/Developer/CommandLineTools/usr/bin/c++)
-- C++ flags: -march=native -mtune=native
-- C++ flags (release): -O3 -Ofast -msse -mfpmath=sse -ffast-math -ftree-vectorize
-- C++ flags (debug): -O0 -g

Training/Evaluation:
--------------------
Batch 600 softmax4 ( loss[soft_cross_entropy]=0.323 metric[categorical_accuracy]=0.189 ) -- 0.021 secs/batch
12.758 secs/epoch
Evaluate with batch size 100
Batch 100 softmax4 ( loss[soft_cross_entropy]=0.319 metric[categorical_accuracy]=0.339 ) --

Memory:
--------
  PID USER      PRI  NI  VIRT   RES S CPU% MEM%   TIME+  Command
12684 mnist_mlp  321.1 00:13.02 8/7  0  19+  258M+ 0B   356K- 12684 1514 running *0[1]     0.00000 0.00000  501 108161+ 115  727+   363+
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