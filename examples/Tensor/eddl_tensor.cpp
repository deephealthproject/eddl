/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <limits>

#include "apis/eddlT.h"

using namespace std;
using namespace eddlT;

int main(int argc, char **argv) {
    int dev = DEV_GPU;

    tensor A=create({10,10});
    
    tensor T=randn({10,10},dev);

    print(T);

    normalize_(T,0,1);

    print(T);

    tensor U=randn({10,3},dev);

    print(U);

    tensor V=mult2D(T,U);

    info(V);

    print(V);


}
