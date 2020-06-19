/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/optimizers/optim.h"

using namespace std;


Optimizer::Optimizer() {
  isshared=false;
  clip_val=-1;
}

void Optimizer::set_clip_val(float v)
{
  clip_val=v;
}

void Optimizer::clip()
{
  if (clip_val<0) return;

  for (int i = 0; i < layers.size(); i++)
    for (int j = 0; j < layers[i]->get_trainable_params_count(); j++)
      layers[i]->gradients[j]->clamp_(-clip_val,clip_val);

}
