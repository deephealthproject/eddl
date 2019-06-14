#include "../../tensor/tensor.h"

void cpu_conv2D(ConvolDescriptor *D);
void cpu_conv2D_grad(ConvolDescriptor *D);
void cpu_conv2D_back(ConvolDescriptor *D);

void cpu_mpool2D(PoolDescriptor*D);
void cpu_mpool2D_back(PoolDescriptor *D);
