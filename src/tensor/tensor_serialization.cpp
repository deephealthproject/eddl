/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <utility>

#include "eddl/tensor/tensor.h"
#include "eddl/hardware/cpu/cpu_tensor.h"
#include "eddl/utils.h"
#include "eddl/helpers.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#endif


// Image stuff
#define STBI_WINDOWS_UTF8

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tensor/stb/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "tensor/stb/stb_image.h"

#define STB_DEFINE
#include "tensor/stb/stb.h"

// Read/Write Numpy
//#include "eddl/tensor/cnpy/cnpy.h"

using namespace std;

// ********* LOAD FUNCTIONS *********
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


Tensor* Tensor::loadfs(std::ifstream &ifs, const string& format) {

    // Choose format
    if (format=="bin") {
        return Tensor::load_from_bin(ifs, 0, -1);
    } else if(format=="onnx"){
        return Tensor::load_from_onnx(ifs);
    } else if(format=="csv" || format=="tsv" || format=="txt"){
        msg("Format deprecated in favor of python: *.'" + format + "'", "Tensor::load");
//        char delimiter;
//        if (format=="csv") {delimiter = ','; }
//        else if (format=="tsv") {delimiter = '\t'; }
//        else { delimiter = ' '; }
//        return Tensor::load_from_txt(ifs, delimiter, 0);
    }else{
        msg("Format not implemented: *.'" + format + "'", "Tensor::load"); // Exits
    }

    return nullptr; // To silent warnings
}

Tensor* Tensor::load_from_bin(std::ifstream &ifs, int start_row, int end_row){
    int r_ndim;

    // Load number of dimensions
    ifs.read(reinterpret_cast<char *>(&r_ndim),  sizeof(int));

    // Load dimensions
    vector<int> r_shape(r_ndim);
    ifs.read(reinterpret_cast<char *>(r_shape.data()), r_ndim * sizeof(int));

    // Compute total size
    int r_size = 1;
    for(int i=0; i<r_ndim; i++){ r_size *= r_shape[i]; }

    // Compute stride
    vector<int> tmp_stride = shape2stride(r_shape);

    // Compute offsets and positions to read
    int start_offset = start_row * tmp_stride[0];
    int n_read;

    if(end_row<0){
        n_read = r_size;
    }else{
        // Compute bytes to read
        int n_rows = end_row - start_row;
        n_read = n_rows * tmp_stride[0];

        // Set new shape
        r_shape[0] = n_rows;

        // Set cursor's position
        ifs.seekg(start_offset*sizeof(float), std::ifstream::cur);
    }

    auto *t1 = new Tensor(r_shape, DEV_CPU);
    ifs.read(reinterpret_cast<char*>(t1->ptr), n_read * sizeof(float));
    // Load content (row-major)
    /*
    auto *r_ptr = new float[r_size];
    ifs.read(reinterpret_cast<char*>(r_ptr), n_read * sizeof(float));

    // Return new tensor
    auto *t1 = new Tensor(r_shape, r_ptr, DEV_CPU);
    */
//    t1->info();
    return t1;
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

        Tensor * temp = new Tensor({t_height, t_width, t_channels}, DEV_CPU);

        // Cast pointer
        // Data in row-major
        t_size = t_width * t_height * t_channels;
        //auto *t_data = new float[t_size];
        float *t_data = temp->ptr;
        for (int i = 0; i < t_size; i++) { t_data[i] = (float) pixels[i]; }

        // Free image
        stbi_image_free(pixels);

        // Re-order components. Data received as HxWxC, and has to be presented as CxHxW
        t = Tensor::permute(temp, {2, 0, 1});
        delete temp;

    } catch(const std::bad_array_new_length &e) {
        msg("There was an error opening the image", "Tensor::load_from_img");
    }

    return t;
}

//Tensor* Tensor::load_from_txt(std::ifstream &ifs, char delimiter, int headerRows){
//    Tensor* t = nullptr;
//    string line;
//    vector<float> values;
//
//    try {
//        CSVIterator it(ifs, delimiter);
//        headerRows = headerRows>=0 ? headerRows : 0;  // Avoid things like -3
//
//        int rows = 0;
//        int cols = it->size();
//
//        // Parse lines
//        for(int i=0; it != CSVIterator(); ++it, ++i){
//            if((i+1)>headerRows){
//                rows++;  // Increment rows
//                for(int j = 0; j < cols; j++){
//                    float cell = std::stof((*it)[j]);
//                    values.push_back(cell);
//                }
//            }else{
//                // If header is present, consume one line
//                // cout << "Ignoring row #" << (i+1) << " as header" << endl;
//            }
//        }
//
//        // Create tensor
//        t = new Tensor({rows, cols});
//        std::copy(std::begin(values), std::end(values), t->ptr);
//
//    } catch(const std::bad_array_new_length &e) {
//        msg("There was an error opening the file", "Tensor::load_from_txt");
//    }
//
//    return t;
//}
//
//Tensor* Tensor::load_from_txt(const string& filename, const char delimiter, int headerRows){
//    Tensor *t = nullptr;
//
//    // Check if file exists (open file stream)
//    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
//    if (!ifs.good()){
//        throw std::runtime_error(std::string("File not found. Check the file name and try again (Tensor::load)"));
//    }
//
//    // Load tensor
//    t = Tensor::load_from_txt(ifs, delimiter, headerRows);
//
//    // Close file stream and return tensor
//    ifs.close();
//    return t;
//}

Tensor* Tensor::load_from_ptr(void * src) {
    char * aux_ptr = (char*) src;

    // Read the number of dimensions
    int ndim = (int) *aux_ptr;
    aux_ptr += sizeof(int);

    // Read dimensions values
    vector<int> shape(ndim);
    memcpy(shape.data(), aux_ptr, ndim * sizeof(int));
    aux_ptr += ndim * sizeof(int);

    // Compute the number of values of data
    int total_size = 1;
    for(int i=0; i < ndim; i++)
        total_size *= shape[i];

    // Read float data
    Tensor * t = new Tensor(shape, DEV_CPU);
    memcpy(t->ptr, aux_ptr, total_size * sizeof(float));

    return t;
}


Tensor* Tensor::load_partial(const string& filename, int start_row, int end_row) {
    // Infer format from filename
    string format = get_extension(filename);

    // Check if file exists (open file stream)
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.good()){
        msg("File not found. Check the file name and try again.", "Tensor::load_partial");
    }

    // Load tensor
    Tensor* t;
    if(format=="bin"){
        t = Tensor::load_from_bin(ifs, start_row, end_row);
    }else{
        msg("Format not implemented: *.'" + format + "'", "Tensor::load");
    }

    // Close file stream and return tensor
    ifs.close();
    return t;
}

// ********* SAVE FUNCTIONS *********
void Tensor::save(const string& filename, string format) {
    // Check if the folder exists
    string folder = filename.substr(0, filename.find_last_of("\\/"));
    if(folder != filename && !pathExists(folder)){
        msg("The file could not be saved. Check if the directory exists or if you have permissions to write in it.", "Tensor::save");
    }

    // Infer format from filename
    if(format.empty()){
        format = get_extension(filename);
    }

    if(format=="png" || format=="bmp" || format=="tga" || format=="jpg" || format=="jpeg" || format=="hdr") { // Images
        save2img(filename, format);
    }else if(format=="bin" || format=="onnx" || format=="csv" || format=="tsv" || format=="txt"){
        // Open file stream, save tensor and close filesteam
        std::ofstream ofs(filename, std::ios::out | std::ios::binary);
        Tensor::savefs(ofs, format);
        ofs.close();
    }else if(format=="npy" || format=="npz"){
        msg("Format deprecated in favor of python: *.'" + format + "'", "Tensor::save");
        //save2numpy(filename, format);
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
    } else if(format=="csv" || format=="tsv" || format=="txt"){
        char delimiter;
        if (format=="csv") {delimiter = ','; }
        else if (format=="tsv") {delimiter = '\t'; }
        else { delimiter = ' '; }
        save2txt(ofs, delimiter, {});
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
    } else if ((this->ndim == 3 && (this->shape[0] < 1 || this->shape[0] > 4)) ||
               (this->ndim == 4 && (this->shape[1] < 1 || this->shape[1] > 4))) {
        msg("3D and 4D tensors must contain a number of channels in the range [1, 4]", "Tensor::save2img");
    } else if (this->ndim == 4 && this->shape[0] != 1) {
        msg("4D tensor must be shaped as (1xCxHxW)", "Tensor::save2img");
    }

    // Clone tensor and copy to CPU
    Tensor *t_cpu = this->clone();
    t_cpu->toCPU();  // Just in case

    // Un/Squeeze dimensions for 2D and 4D
    if (t_cpu->ndim == 4){ // CxHxW
        t_cpu->squeeze_();  // Careful with: 1x3x32x1 => 3x32
    }
    if(t_cpu->ndim == 2){ // 1xHxW
        t_cpu->unsqueeze_();
    }

    // Re-order components. From CxHxW  => HxWxC
    Tensor *t = Tensor::permute(t_cpu, {1, 2, 0});  // Performs clone
    delete t_cpu;

    // Normalize image (for RGB must fall between 0 and 255) => Not a good idea
    //t->normalize_(0.0f, 255.0f);

    // Cast pointer
    // Data in row-major!!!
    auto* data= new uint8_t[t->size];
    for(int i=0;i<t->size;i++){ data[i]=t->ptr[i]; }

    // Components
    int height = t->shape[0];
    int width = t->shape[1];
    int channels = t->shape[2];

    // Save image
    if(format=="png") {
        stbi_write_png(filename.c_str(), width, height, channels, data, width * channels);  // width x channels
    }else if(format=="bmp"){
        stbi_write_bmp(filename.c_str(), width, height, channels, data);
    }else if(format=="tga"){
        stbi_write_tga(filename.c_str(), width, height, channels, data);
    }else if(format=="jpg" || format=="jpeg"){
        stbi_write_jpg(filename.c_str(), width, height, channels, data, 100);
//    }else if(format=="hdr"){
//        stbi_write_hdr(filename.c_str(), this->shape[3], this->shape[2], this->shape[1], data);
    }else{
        msg("Format not implemented", "Tensor::save2img");
    }

    delete[] data;
    delete t;
}

//void Tensor::save2numpy(const string &filename, string format){
//    vector<size_t> t_shape;
//    for(auto &s : this->shape){
//        t_shape.push_back(s);
//    }
//    cnpy::npy_save(filename, this->ptr, t_shape, "w");
//}

void Tensor::save2txt(std::ofstream &ofs, const char delimiter, const vector<string> &header){
    if(this->ndim!=2){
        msg("This method is only valid for tensors with 2 dimensions", "Tensor::save2txt");
    }

    // Write file
    if (ofs.is_open()) {

        // Write header
        for(int i=0; i<header.size(); i++){
            ofs << header[i];

            if(i==header.size()-1){ ofs << endl; }
            else{ ofs << delimiter; }
        }

        // Write content
        for(int i = 0; i<this->size; i++){
            ofs << this->ptr[i];

            // One line per row
            if((i+1) % this->shape[1]==0){ ofs << endl; }
            else{ ofs << delimiter; }
        }

    }
}

void Tensor::save2txt(const string& filename, const char delimiter, const vector<string> &header){
    // Check if the folder exists
    string folder = filename.substr(0, filename.find_last_of("\\/"));
    if(folder != filename && !pathExists(folder)){
        msg("The file could not be saved. Check if the directory exists or if you have permissions to write in it.", "Tensor::save");
    }

    // Open file stream
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);

    // Save
    this->save2txt(ofs, delimiter, header);

    // Close file stream
    ofs.close();
}

std::pair<void*, size_t> Tensor::save2ptr() {

    Tensor * aux_tensor;
    bool copied;
    // Check the tensor device to make a copy or not
    if(!this->isCPU()){
        aux_tensor = new Tensor(this->shape, DEV_CPU);
        Tensor::copy(this, aux_tensor);
        copied = true;
    } else {
        aux_tensor = this;
        copied = false;
    }

    // Reserve memory for: ndims value, shape vector and tensor float data
    size_t needed_mem = sizeof(int) + (aux_tensor->shape.size() * sizeof(int)) + (aux_tensor->size * sizeof(float));
    float * dest = get_fmem(needed_mem, "save2ptr()");
    char * aux_ptr = (char*) dest;

    // Store the number of dimensions
    memcpy(aux_ptr, &aux_tensor->ndim, sizeof(int));
    aux_ptr += sizeof(int);

    // Store the dimensions values
    memcpy(aux_ptr, aux_tensor->shape.data(), aux_tensor->shape.size() * sizeof(int));
    aux_ptr += aux_tensor->shape.size() * sizeof(int);

    // Store the float data
    memcpy(aux_ptr, aux_tensor->ptr, aux_tensor->size * sizeof(float));

    if(copied) delete aux_tensor; // Delete the auxiliary copy in CPU

    return std::make_pair(dest, needed_mem);
}
