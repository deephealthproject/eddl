/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <utility>

#include "tensor.h"
#include "../hardware/cpu/cpu_hw.h"
#include "../utils.h"

#ifdef cGPU
#include "../hardware/gpu/gpu_tensor.h"
#include "../hardware/gpu/gpu_hw.h"
#include "../hardware/gpu/nn/gpu_nn.h"
#endif


// Image stuff
#define STBI_WINDOWS_UTF8

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_DEFINE
#include "stb/stb.h"

// Read/Write Numpy
#include "cnpy/cnpy.h"

using namespace std;

// ********* LOAD FUNCTIONS *********
Tensor* Tensor::load_from_csv(const string & fname)
{
  FILE *fe = fopen(fname.c_str(), "rt");
  if (fe == nullptr) {
      fprintf(stderr, "%s not found\n", fname.c_str());
      exit(1);
  }

  vector<int> shape;
  int ndim,v;

  fscanf(fe,"%d",&ndim);
  for(int i=0;i<ndim;i++) {
    fscanf(fe,"%d",&v);
    shape.push_back(v);
  }

  Tensor *n=new Tensor(shape,DEV_CPU);

  for (int i = 0; i < n->size; ++i)
      fscanf(fe,"%f ",&(n->ptr[i]));

  return n;
}

Tensor* Tensor::load(const string& filename, string format){
    // Infer format from filename
    if(format.empty()){
        format = get_extension(filename);
    }

    // Check source type
    if(format=="npy" || format=="npz"){
        msg("Numpy files need a source type to be specified: 'Tensor::loadt<type>(filename)'");
    }

    // Default type to be ignored
    // Ignore IDE warnings (some times they have problems with templates)
    return Tensor::load<float>(filename, std::move(format));
}


Tensor* Tensor::loadfs(std::ifstream &ifs, string format) {

    // Choose format
    if(format=="bin") {
        return Tensor::load_from_bin(ifs);
    } else if(format=="onnx"){
        return Tensor::load_from_onnx(ifs);
    }else{
        msg("Format not implemented: *.'" + format + "'", "Tensor::load");
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


Tensor* Tensor::load_from_img(const string &filename, const string &format){
    Tensor* t = nullptr;

    try {
        int t_width, t_height, t_channels, t_size;

        // IMPORTANT! There might be problems if the image is grayscale, a png with 3 components,...
        // Set number of channels to read
        unsigned char *pixels = stbi_load(filename.c_str(), &t_width, &t_height, &t_channels, STBI_default);

        // Cast pointer
        t_size = t_width * t_height * t_channels;
        auto *t_data = new float[t_size];
        for (int i = 0; i < t_size; i++) { t_data[i] = (float) pixels[i]; }

        // Free image
        stbi_image_free(pixels);

        // Re-order components. Data received as 1xWxHxC, and has to be presented as 1xCxHxW
        t = new Tensor({1, t_width, t_height, t_channels}, t_data, DEV_CPU);
        t = Tensor::permute(t, {0, 3, 2, 1});

    } catch(const std::bad_array_new_length &e) {
        msg("There was an error opening the image", "Tensor::load_from_img");
    }

    return t;
}

// ********* SAVE FUNCTIONS *********
void Tensor::save(const string& filename, string format) {
    // Infer format from filename
    if(format.empty()){
        format = get_extension(filename);
    }

    if(format=="png" || format=="bmp" || format=="tga" || format=="jpg" || format=="jpeg" || format=="hdr") { // Images
        save2img(filename, format);
    }else if(format=="bin" || format=="onnx"){
        // Open file stream
        std::ofstream ofs(filename, std::ios::out | std::ios::binary);

        // Save
        Tensor::savefs(ofs, format);

        // Close file stream
        ofs.close();
    }else if(format=="npy" || format=="npz"){
        save2numpy(filename, format);
    }else{
        msg("Format not implemented: *.'" + format + "'", "Tensor::save");
    }
}

void Tensor::savefs(std::ofstream &ofs, string format) {
    if (!isCPU()){
        msg("Only save CPU Tensors", "Tensor::save");
    }

    // Choose format
    if(format=="bin") {
        save2bin(ofs);
    } else if(format=="onnx"){
        save2onnx(ofs);
    }else{
        msg("Format not implemented: *.'" + format + "'", "Tensor::save");
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


void Tensor::save2img(const string& filename, string format){
    if (this->ndim < 2 || this->ndim > 4){
        msg("Tensors should be 2D (HxW), 3D (CxHxW) or 4D (1xCxHxW)","Tensor::save2img");
    } else if ((this->ndim == 3 || this->ndim == 4) && (this->ndim < 1 || this->ndim > 4)) {
        msg("3D and 4D tensors must contain a number of channels in the range [1, 4]","Tensor::save2img");
    } else if (this->ndim == 4 && this->shape[0] != 1) {
        msg("4D tensor must be shaped as (1xCxHxW)","Tensor::save2img");
    }

    // Clone tensor and copy to CPU
    Tensor *t = this->clone();
    t->toCPU();  // Just in case

    // Un/Squeeze dimensions for 2D and 4D
    if (t->ndim == 4){ // CxHxW
        t->squeeze_();  // Careful with: 1x3x32x1 => 3x32
    }
    if(t->ndim == 2){ // 1xHxW
        t->unsqueeze_();
    }

    // Re-order components. From CxHxW  => WxHxC
    t = Tensor::permute(t, {2, 1, 0});  // Performs clone

    // Normalize image (for RGB must fall between 0 and 255) => Not a good idea
    t->normalize_(0.0f, 255.0f);

    // TODO: I don't see the need to cast this (but if i remove it, it doesn't work)
    // Cast pointer
    auto* data= new uint8_t[t->size];
    for(int i=0;i<t->size;i++){ data[i]=t->ptr[i]; }

    // Save image
    if(format=="png") {
        stbi_write_png(filename.c_str(), t->shape[0], t->shape[1], t->shape[2], data, t->shape[0] * t->shape[2]);  // width x channels
    }else if(format=="bmp"){
        stbi_write_bmp(filename.c_str(), t->shape[0], t->shape[1], t->shape[2], data);
    }else if(format=="tga"){
        stbi_write_tga(filename.c_str(), t->shape[0], t->shape[1], t->shape[2], data);
    }else if(format=="jpg" || format=="jpeg"){
        stbi_write_jpg(filename.c_str(), t->shape[0], t->shape[1], t->shape[2], data, 100);
//    }else if(format=="hdr"){
//        stbi_write_hdr(filename.c_str(), this->shape[3], this->shape[2], this->shape[1], data);
    }else{
        msg("Format not implemented", "Tensor::save2img");
    }

}

void Tensor::save2numpy(const string &filename, string format){
    vector<size_t> t_shape;
    for(auto &s : this->shape){
        t_shape.push_back(s);
    }
    cnpy::npy_save(filename, this->ptr, t_shape, "w");
}
