//
// Created by Salva Carrión on 2019-05-16.
//

#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer.h"

using namespace std;

int LTranspose::total_layers = 0;

// ---- CONVOLUTION ----
LTranspose::LTranspose(Layer *parent, const initializer_list<int> &dims, string name, int dev) : LinLayer(name, dev) {
    // TODO: Implement
    total_layers++;
    this->dims = dims;
}