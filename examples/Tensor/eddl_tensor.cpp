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

#include "apis/eddl.h"
#include "apis/eddlT.h"

using namespace std;

int main(int argc, char **argv) {
    int dev = DEV_CPU;


    tensor T=eddlT::randn({10,10},dev);

    eddlT::print(T);

    eddlT::normalize_(T,0,1);

    eddlT::print(T);


}
