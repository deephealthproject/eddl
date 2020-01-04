/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "initializer.h"

using namespace std;

/**
 * Initializer that generates the identity matrix.
 *
 * Only use for 2D matrices. If desired matrix is not square, gets padded with zeros for the additional rows/columns.
 *
 * @param gain float; Multiplicative factor to apply to the identity matrix.
*/
IIdentity::IIdentity(float gain) : Initializer("identity") {
    // Todo: Implement
    this->gain = gain;
}
void IIdentity::apply(Tensor* params)
{
  msg("Identity not implemented","IIdentity::apply");    
}
