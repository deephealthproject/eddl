/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include "tensor.h"
#include "../hardware/cpu/cpu_hw.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"

#ifdef cGPU
#include "../hardware/gpu/gpu_tensor.h"
#include "../hardware/gpu/gpu_hw.h"
#include "../hardware/gpu/nn/gpu_nn.h"
#endif


using namespace std;

// ********* LOAD FUNCTIONS *********
Tensor* Tensor::load(const string& filename, const string& format) {

    if(format=="png") { // Images
        return Tensor::load_from_img(filename, format);
    }else if(format=="bin" || format=="onnx"){
        // Open file stream
        std::ifstream ifs(filename, std::ios::in | std::ios::binary);

        // Save
        Tensor* t = Tensor::loadfs(ifs, format);

        // Close file stream
        ifs.close();

        return t;
    }else{
        msg("Not implemented", "Tensor::save");
    }
}

Tensor* Tensor::loadfs(std::ifstream &ifs, const string& format) {

    // Choose format
    if(format=="bin") {
        return Tensor::load_from_bin(ifs);
    } else if(format=="onnx"){
        return Tensor::load_from_onnx(ifs);
    }else{
        msg("Not implemented", "Tensor::load");
    }

}

Tensor* Tensor::load_from_bin(std::ifstream &ifs){
    int r_ndim;

    // Load number of dimensions
    ifs.read(reinterpret_cast<char *>(&r_ndim),  sizeof(int));

    // Load dimensions
    vector<int> r_shape(r_ndim);
    ifs.read(reinterpret_cast<char *>(r_shape.data()), r_ndim * sizeof(int));

    // Compute total size
    int r_size = 1;
    for(int i=0; i<r_ndim; i++){ r_size *= r_shape[i]; }

    // Load content (row-major)
    auto *r_ptr = new float[r_size];
    ifs.read(reinterpret_cast<char*>(r_ptr), r_size * sizeof(float));

    // Return new tensor
    return new Tensor(r_shape, r_ptr, DEV_CPU);
}

Tensor* Tensor::load_from_onnx(std::ifstream &ifs){
    msg("Not implemented", "Tensor::load_from_onnx");

    // Return new tensor
    return new Tensor();
};

Tensor* Tensor::load_from_img(const string &filename, const string& format){
    msg("Not implemented", "Tensor::load_from_img");

    // Return new tensor
    return new Tensor();
}


// ********* SAVE FUNCTIONS *********
void Tensor::save(const string& filename, const string& format) {

    if(format=="png") { // Images
        save2img(filename, format);
    }else if(format=="bin" || format=="onnx"){
        // Open file stream
        std::ofstream ofs(filename, std::ios::out | std::ios::binary);

        // Save
        Tensor::savefs(ofs, format);

        // Close file stream
        ofs.close();
    }else{
        msg("Not implemented", "Tensor::save");
    }
}

void Tensor::savefs(std::ofstream &ofs, const string& format) {
    if (!isCPU()){
        msg("Only save CPU Tensors", "Tensor::save");
    }

    // Choose format
    if(format=="bin") {
        save2bin(ofs);
    } else if(format=="onnx"){
        save2onnx(ofs);
    }else{
        msg("Not implemented", "Tensor::save");
    }

}


void Tensor::save2bin(std::ofstream &ofs){

    // Save number of dimensions
    ofs.write(reinterpret_cast<const char *>(&this->ndim), sizeof(int));

    // Save dimensions
    ofs.write(reinterpret_cast<const char *>(this->shape.data()), this->shape.size() * sizeof(int));

    // Save content (row-major)
    ofs.write(reinterpret_cast<const char *>(this->ptr), this->size * sizeof(float));
}

void Tensor::save2onnx(std::ofstream &ofs){
    msg("Not implemented", "Tensor::save2onnx");
};


void Tensor::save2img(const string& filename, const string& format){
    if (this->ndim!=4) {
        msg("Tensors should be 4D: 1xCxHxW","save_png");
    }

    // Re-order axis
    Tensor *t = this->permute({0, 2, 3, 1}); // Data must be presented as WxHxC => [(ARGB), (ARGB), (ARGB),...]
    t->ToCPU();  // Just in case

    // Normalize image (for RGB must fall between 0 and 255)
    t->normalize_(0.0f, 255.0f);

    // TODO: I don't see the need to cast this (but if i remove it, it doesn't work)
    // Cast pointer
    auto* data= new uint8_t[this->size];
    for(int i=0;i<this->size;i++){ data[i]=t->ptr[i]; }

    // Save image
    if(format=="png"){
        //  // (w, h, c, data, w*channels) // w*channels => stride_in_bytes
        stbi_write_png(filename.c_str(), this->shape[3], this->shape[2], this->shape[1], data, this->shape[3] * this->shape[1]);
    }else{
        msg("Format not implemented", "Tensor::save2img");
    }

}
