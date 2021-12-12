/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <iterator>
#include <fstream>  // for the linux stuff
#include <iostream>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <new>      // included for std::bad_alloc
#include <string>
#include <limits>
#include <algorithm>
#include <sstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <iomanip>
#include <limits>



#include "eddl/system_info.h"
#include "eddl/utils.h"
#include "eddl/profiling.h"

#ifdef EDDL_LINUX
#include "sys/mman.h"
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

#ifdef EDDL_APPLE
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>
#endif

#ifdef EDDL_WINDOWS
#include <windows.h>
#endif


void msg(const string& text, const string& title){
    string s(text);
    if(!title.empty()){
        s += " (" + title + ")";
    }
    std::cerr << "==================================================================\n";
    std::cerr << "⚠️  " << s << " ⚠️"<<std::endl;
    std::cerr << "==================================================================\n\n";

    throw std::runtime_error("RuntimeError: " + title);
}

void set_text_green(){
  printf("\033[0;32m");
}

void set_text_red(){
  printf("\033[0;31m");
}

void set_text_default(){
  printf("\033[0m");
}

void * eddl_malloc(size_t size, const string & str_info){
    constexpr size_t alignment_block_size = 64;

    // Careful with memory overcommitment:
    // https://stackoverflow.com/questions/48585079/malloc-on-linux-without-overcommitting
    // TODO: This function does not work properly (...but it does, at least most of the time -for linux and mac-)
    void * ptr = nullptr;
    bool error = false;
    int rc = -1;


#if defined(EDDL_LINUX) || defined(EDDL_APPLE)
    // Check if free memory is bigger than requested
    /* get_free_mem() is not actually necessary,
       memory allocating system calls set the 'errno' variable
       to indicate which was the error when trying to allocate
       the CPU memory

    unsigned long int freemem = get_free_mem();
    error = (size > freemem);
    */

    if (! error) {

        // New vs Malloc *******************
        // New is the C++ way of doing it
        // New is type-safe, Malloc is not
        // New calls your type constructor, Malloc not - Same for destructor
        // New is an operator, Malloc a function (slower)
        try {
            //ptr = new float[size];
            //ptr=(float *)malloc(size*sizeof(float));
            //ptr=(float *)aligned_alloc(64, size*sizeof(float));
            rc = posix_memalign((void **)&ptr, alignment_block_size, size);
            error = (0 != rc);
            errno = rc;
        }
        catch (std::bad_alloc & badAlloc) { error = true; }
    }

#elif defined(EDDL_WINDOWS)
    errno = 0;
    ptr = _aligned_malloc(size, alignment_block_size);
    error = (nullptr == ptr || errno == ENOMEM);
#else
#error "A proper configuration must define either EDDL_LINUX, EDDL_APPLE or EDDL_WINDOWS"
#endif

    // Check for errors
    // Not enough free memory
    if (error || ptr == nullptr) {
        if (ptr != nullptr) eddl_free(ptr);
        //throw std::runtime_error("Error allocating " + string(bytes2human(size * sizeof(float))) + " in " + string(str));
        throw std::runtime_error("Error " + std::to_string(errno)
                                + " allocating " + string(bytes2human(size, 0)) + " bytes at "
                                + string(__FILE__) + "(" + std::to_string(__LINE__) + ") " + str_info);
    }

    return ptr;
}

void eddl_free(void * ptr)
{
#if defined(EDDL_LINUX) || defined(EDDL_APPLE)
    free(ptr);
#elif defined(EDDL_WINDOWS)
    _aligned_free(ptr);
#else
#error "A proper configuration must define either EDDL_LINUX, EDDL_APPLE or EDDL_WINDOWS"
#endif
}

float *get_fmem(unsigned long int size, const string &str)
{
    return (float *)eddl_malloc(size * sizeof(float), str);
}


string bytes2human(unsigned long long int bytes, int decimals){
    vector<string> prefix = {"B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"};
    double size = 0;
    int i = 0;

    if(bytes>0){ // bytes == 0 is a special case since log(0) is undefined
        // Get prefix
        i = floor(log(bytes) / log(1024));

        // Compute size
        size = (float)bytes / pow(1024, i);
    }

    // Format string
    if(i==0){ decimals = 0; } // Make "Bytes" integers
    std::stringstream stream;
    stream << std::fixed << std::setprecision(decimals) << size;
    stream << prefix[i]; // Prefix
    std::string s = stream.str();

    return s;
}


#ifdef EDDL_LINUX
unsigned long get_free_mem() {
        std::string token;
        std::string type = "MemAvailable:";
        std::ifstream file("/proc/meminfo");
        while(file >> token) {
            if(token == type) {
                unsigned long mem;
                if(file >> mem) {
                    return mem * 1024; // From kB to Bytes
                } else {
                    return 0;
                }
            }
            // ignore rest of the line
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        return 0; // nothing found
    }
#endif

#ifdef EDDL_APPLE
unsigned long get_free_mem() {
    // TODO: Review. This doesn't work correctly
    mach_port_t host_port;
    mach_msg_type_number_t host_size;
    vm_size_t pagesize;
    host_port = mach_host_self();
    host_size = sizeof(vm_statistics64) / sizeof(integer_t);
    host_page_size(host_port, &pagesize);

    struct vm_statistics64 vm_stat{};
    if (host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size) != KERN_SUCCESS) {
        throw std::invalid_argument("Failed to fetch vm statistics");
    }
    unsigned long mem_free = (vm_stat.free_count +vm_stat.inactive_count) * pagesize;
    //fprintf(stderr,"%s Free\n",bytes2human(mem_free));

    return mem_free;
}

#endif

#ifdef EDDL_WINDOWS
unsigned long get_free_mem() {
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return -1;
}
#endif

string get_extension(string filename){
    std::string::size_type idx = filename.rfind('.');
    if(idx != std::string::npos){
        return filename.substr(idx+1);
    }
    return "";
}

vector<vector<int>> parse_indices(vector<string> str_indices, const vector<int>& shape){
    string delimiter(":");
    vector<vector<int>> ranges;

    // Shapes must match
    if(str_indices.size() != shape.size()){
        int diff = shape.size() - str_indices.size();
        if(diff>=0){
            for(int i=0; i<diff; i++){
                str_indices.emplace_back(":");
            }
        }else{
            msg( "The number of dimensions of the indices cannot be greater than the shape of the tensor to match", "utils::parse_indices");
        }
    }

    // Parse string indices
    for(int i=0; i<str_indices.size(); i++){
        int min, max;

        // Remove whitespaces
        string str = str_indices[i];
        std::string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
        str.erase(end_pos, str.end());

        // Find delimiters
        int pos = str.find(delimiter);
        if(pos != string::npos){ // Found
            if(str==delimiter){  // ":"
                min = 0;
                max = shape[i];
            }else{
                if (pos==0){ // ":5"
                    min = 0;
                    max = std::stoi(str.substr(pos+delimiter.length(), string::npos));  // Numpy style
                }else if(pos==str.length()-1){  // "5:"
                    min = std::stoi(str.substr(0, str.length()-delimiter.length()));
                    max = shape[i];
                }else{  // "5:10"
                    min = std::stoi(str.substr(0, pos - 0));  // (start_pos, len= end_pos-start_pos)
                    max = std::stoi(str.substr(pos+delimiter.length(), string::npos));  // Numpy style
                }
            }

            max -= 1;  // last index is not included
        }else{  // Not found => "5"
            min = std::stoi(str);
            max = min;
        }
        // Negative indices // len + (-x)
        if(min<0) { min = shape[i] + min; }
        if(max<0) { max = shape[i] + max; }

        ranges.push_back({min, max});
    }


    // Second check (negative values, max < min, or max > shape)
    for(int i=0; i<ranges.size(); i++){
        string common_str = "Invalid indices: '" + str_indices[i] + "'. ";
        if(ranges[i][0] < 0 || ranges[i][1] < 0){
            msg( common_str + "Indices must be greater than zero.", "utils::parse_indices");
        }else if(ranges[i][1] < ranges[i][0]){
            msg(common_str + "The last index of the range must be greater or equal than the first.", "utils::parse_indices");
        } else if(ranges[i][1] >= shape[i]){
            msg(common_str + "The last index of the range must fit in its dimension.", "utils::parse_indices");
        }
    }
    return ranges;
}

vector<int> indices2shape(vector<vector<int>> ranges){
    vector<int> shape;
    for(auto & range : ranges){
        shape.push_back(range[1]-range[0]+1);
    }
    return shape;
}

int shape2size(vector<int> shape){
    int size = 1;
    for(int i=0; i<shape.size(); i++){
        size *= shape[i];
    }
    return size;
}

vector<int> shape2stride(const vector<int>& shape){
    vector<int> stride = {1};

    for(int i=shape.size()-1; i>0; i--){
        int s = shape[i];
        int s2 = stride[0];
        stride.insert(stride.begin(), s*s2);
    }

    return stride;
}

vector<int> permute_shape(const vector<int>& ishape, const vector<int>& dims){
    vector<int> oshape;
    if(dims.size()!=ishape.size()){
        msg("Dimensions do not match", "utils::permute_indices");
    }else{
        for(auto &d : dims){
            oshape.emplace_back(ishape[d]);
        }
    }

    return oshape;
}

int* permute_indices(const vector<int>& ishape, const vector<int>& dims){
    int* addresses = nullptr;
    vector<int> oshape = permute_shape(ishape, dims);

    // Compute size
    int isize = shape2size(ishape);
    int osize = shape2size(oshape);

    // Check if the shapes are compatible
    if (ishape.size() != oshape.size() || isize!=osize){
        msg("Incompatible dimensions", "utils::permute_indices");
    }else{
        vector<int> istride = shape2stride(ishape);
        vector<int> ostride = shape2stride(oshape);
        addresses = new int[isize];

        // For each output address (0,1,2,3,...n), compute its indices
        // Then add the minimum of each range, and compute the raw address
        for(int i=0; i<isize; i++) {

            // Extract indices
            int B_pos = 0;
            for(int d=0; d<ishape.size(); d++){
                // Compute output indices at dimension d, but permuted
                int A_idx = (i/istride[dims[d]]) % ishape[dims[d]];  // (52 / 32) % 32=> [1, 20]
                B_pos += A_idx * ostride[d];
            }

            // Save address translation
            addresses[B_pos] = i;
        }
    }

    return addresses;  // Be careful! It's easy to forget about this pointer and have a memory leak
}

int* ranges2indices(vector<int> ishape, vector<vector<int>> ranges){
    // Returns an array with the linear positions of the ranges to perform fast translations
    // [0:2, 5] {H=10, W=7}=> ([0,1], [5]) => (0*7+5),(1*7)+5,...

    // Compute output dimensions
    vector<int> istride = shape2stride(ishape);

    vector<int> oshape = indices2shape(ranges);
    vector<int> ostride = shape2stride(oshape);
    int osize = shape2size(oshape);
    int* addresses = new int[osize];  // Because the batch is 1 (default), then it's resized

    // For each output address (0,1,2,3,...n), compute its indices
    // Then add the minimum of each range, and compute the raw address
    for(int i=0; i<osize; i++) {

        // Extract indices
        int A_pos = 0;
        for(int d=0; d<ranges.size(); d++){
            // Compute output indices at dimension d
            int B_idx = (i/ostride[d]) % oshape[d];  // (52 / 32) % 32=> [1, 20]

            // Compute input indices at dimension d
            int A_idx = B_idx + ranges[d][0];  // B_index + A_start => [0, 0, 0] + [0, 5, 5]
            A_pos += A_idx * istride[d];
        }

        // Save address translation
        addresses[i] = A_pos;
    }

    return addresses;  // Be careful! It's easy to forget about this pointer and have a memory leak
}

vector<int> expand_shape(const vector<int>& ishape, int size){
    vector<int> new_shape;

    // Check if there are dimensions to expand
    bool willExpand = false;
    for(auto &d : ishape){
        if (d!=1){
            new_shape.push_back(d);
        }else{
            willExpand = true;
            new_shape.push_back(size);
        }
    }

    // Check if it can be expanded
    if(!willExpand){
        std::cerr << "This tensor cannot be expanded. At least one dimension of size 1 is required. " << "(Tensor::expand)" << std::endl;
    }

    return new_shape;
}

int* expand_indices(const vector<int>& ishape, int size){
    int* addresses = nullptr;
    vector<int> oshape = expand_shape(ishape, size);

    // Compute size
    int isize = shape2size(ishape);
    int osize = shape2size(oshape);

    vector<int> istride = shape2stride(ishape);
    vector<int> ostride = shape2stride(oshape);
    addresses = new int[osize];

    // For each output address (0,1,2,3,...n), compute its indices in input
    // Then add the minimum of each range, and compute the raw address
    for(int i=0; i<osize; i++) {

        // Extract indices
        int A_pos = 0;
        int B_pos = 0;
        for(int d=0; d<oshape.size(); d++){
            // Compute output indices at dimension d
            int B_idx = (i/ostride[d]) % oshape[d];  // (52 / 32) % 32=> [1, 20]
            int A_idx;

            // Translate to input
            if (ishape[d]==1){ // Dimension to be expanded
                A_idx = 0;
            }else{
                A_idx = B_idx;
            }

            // Compute partial pointers
            A_pos += A_idx * istride[d];
            B_pos += B_idx * ostride[d];
        }

        // Save address translation
        addresses[i] = A_pos;
    }

    return addresses;  // Be careful! It's easy to forget about this pointer and have a memory leak
}

vector<int> getBroadcastShape(vector<int> shape1, vector<int> shape2){
    // Normalize broadcast shape: (3)*(1,3), (3)*(6,2,5,3,5), (5, 3)*(5, 3),...
    vector<int> broadcast_shape;

    // Check dimensions
    if (shape1.size() == shape2.size()){  // Same shape
        broadcast_shape = shape2;

    }else if(shape1.size()==1){  // Shape1 has 1 dimension
        // Broadcast: [3] AND [12, 3, 7, 7] => [1, 3, 1, 1]
        for(int i = 0; i < shape2.size(); ++i) {
            if (shape1[0] == shape2[i]) {
                broadcast_shape.push_back(shape2[i]);
            }else{ broadcast_shape.push_back(1); }
        }
    }else{
        // None
    }

    return broadcast_shape;
}

vector<int> getTilesRepetitions(const vector<int>& broadcast_from, const vector<int>& broadcast_to){
    // Compute repetitions to perform a given broadcast: (3, 1, 1)*(3, 28, 28) => (1, 28, 28)
    vector<int> tile_repetitions;  // Tile repetitions

    // Check dimensions
    if(broadcast_from.size()!=broadcast_from.size()){
        msg("'broadcast_from' and 'broadcast_to' must have the same number of dimensions", "utils::getTilesRepetitions");
    }

    // Compute repetitions
    for(int i=0; i<broadcast_to.size(); i++){
        if(broadcast_from[i]==broadcast_to[i]) { // 3=>3: No repeat
            tile_repetitions.push_back(1);
        }else if(broadcast_from[i]==1){ // 1=>5: Repeat 5 times
            tile_repetitions.push_back(broadcast_to[i]);
        }else{
            // Error
        }
    }

    return tile_repetitions;
}

bool is_number(const std::string& s){
    return !s.empty() && std::find_if(s.begin(), s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
}

bool pathExists(const std::string &s){
    struct stat buffer;
    return (stat (s.c_str(), &buffer) == 0);
}

string get_parent_dir(const string& fname){
    size_t pos = fname.find_last_of("\\/");
    return (std::string::npos == pos)
           ? ""
           : fname.substr(0, pos);
}

string replace_str(const string& value, const string& oldvalue, const string& newvalue){
    string new_value = string(value);

    size_t index = 0;
    while (true) {
        /* Locate the substring to replace. */
        index = new_value.find(oldvalue, index);
        if (index == std::string::npos) break;

        /* Make the replacement. */
        new_value.replace(index, newvalue.length(), newvalue);

        /* Advance index forward so the next iteration doesn't pick it up as well. */
        index += newvalue.length();
    }

    return new_value;
}

string normalize_layer_name(const string& value){
    string new_value = string(value);
    new_value = replace_str(new_value, "/", "_");
    new_value = replace_str(new_value, "-", "_");
    new_value = replace_str(new_value, ":", "_");
    return new_value;
}

vector<int> compute_squeeze(vector<int> shape, int axis, bool ignore_batch){
    int faxis = axis+(int)ignore_batch;
    int lastdim = (int)(shape.size()-1);

    // Check dimension bounds
    if (faxis > lastdim) {
        msg("Number of dimensions exceeded (" + to_string(axis) + " >= " + to_string((int)(shape.size()-(int)ignore_batch)) + ")", "compute_squeeze");
    }else  if (faxis < -1){
        msg("The axis must be greater or equal than zero; or -1 (special case)", "compute_squeeze");
    }

    // Remove single dimension entries from the array
    vector<int> new_shape;

    // Ignore batch if needed
    if(ignore_batch){
        new_shape.push_back(shape[0]);
    }

    for(int i=(int)ignore_batch; i<shape.size(); i++){
        int dim = shape[i];

        // If dimension is greater than 1 or batch is ignored
        if((dim>1) || (i!=faxis && axis!=-1)){
            new_shape.push_back(dim);
        }
    }

    return new_shape;
};

vector<int> compute_unsqueeze(vector<int> shape, int axis, bool ignore_batch){
    int faxis = axis+(int)ignore_batch;

    // Check dimension bounds
    if (faxis > shape.size()) {
        msg("Number of dimensions exceeded (" + to_string(axis) + " >= " + to_string((int)(shape.size()-(int)ignore_batch)) + ")", "compute_unsqueeze");
    }else  if (faxis < 0){
        msg("The axis must be greater or equal than zero", "compute_unsqueeze");
    }

    vector<int> new_shape(shape);
    new_shape.insert(new_shape.begin()+faxis, 1); // Add one dimension to the beginning
    return new_shape;
}


vector<int> address2indices(int address, const vector<int>& shape, const vector<int>& strides){
    // Check sizes
    if(shape.size()!=strides.size()){
        msg("Shape and strides must have the same size", "utils::address2indices");
    }

    // Compute size
    int tsize = 1;
    for(auto &s : shape) { tsize *= s; }

    // Check maximum size
    if(address > tsize-1){
        msg("The address cannot greater than the maximum possible address with the given shape", "utils::address2indices");
    }

    // Reserve memory
    vector<int> indices;
    int ndim = strides.size();
    indices.reserve(ndim);

    // Compute indices
    fast_address2indices(address, indices.data(), shape.data(), strides.data(), ndim);

    return indices;
}

unsigned int indices2address(const vector<int>& indices, const vector<int>& strides){
    // Check sizes
    if(indices.size()!=strides.size()){
        msg("Indices and strides must have the same size", "utils::indices2address");

    }
    // Compute address
    int address = fast_indices2address(indices.data(), strides.data(), indices.size());

    return address;
}

bool isPaddingAsymmetric(vector<int> padding){
    // Check if padding is even
    if(padding.size()%2!=0){
        msg("'padding' must have an even number of elements");
    }

    // Check for asymmetric paddings
    for(int i=0; i<padding.size(); i+=2){
        if(padding[i]!=padding[i+1]){
            return true;
        }
    }
    return false;
}

vector<vector<int>> cartesian_product(const vector<vector<int>>& vectors){
    vector<vector<int>> results = {{}};
    for (auto &vec : vectors){ // Vectors: {0, 1}, {5, 6, 7}, {8, 9}
        vector<vector<int>> temp;

        for(auto &res : results){  // Previous solution: {{0}, {1}}
            for(auto &elem : vec){  // Elements 1, 2, 3,...
                vector<int> new_vec = res;
                new_vec.push_back(elem);
                temp.push_back(new_vec);
            }
        }
        results.clear();
        results = temp;
    }
    return results;
}

WrappingMode getWrappingMode(string mode){
    if(mode == "constant"){
        // (k k k k | a b c d | k k k k)
        // The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
        return WrappingMode::Constant;
    }else if(mode == "reflect"){
        // (d c b a | a b c d | d c b a)
        // The input is extended by reflecting about the edge of the last pixel.
        return WrappingMode::Reflect;
    }else if(mode == "nearest"){
        // (a a a a | a b c d | d d d d)
        // The input is extended by replicating the last pixel.
        return WrappingMode::Nearest;
    }else if(mode == "mirror"){
        // (d c b | a b c d | c b a)
        // The input is extended by reflecting about the center of the last pixel.
        return WrappingMode::Mirror;
    }else if(mode == "wrap"){
        // (a b c d | a b c d | a b c d)
        // The input is extended by wrapping around to the opposite edge.
        return WrappingMode::Wrap;
    }else if(mode == "original"){
        // (o o o o | a b c d | o o o o)
        // The input is extended by filling all values beyond the edge with the original values
        return WrappingMode::Original;
    }else {  // constant
        return WrappingMode::Constant;
    }
}

TransformationMode getTransformationMode(string mode){
    if(mode == "half_pixel"){
        return TransformationMode::HalfPixel;
    }else if(mode == "pytorch_half_pixel"){
        return TransformationMode::PytorchHalfPixel;
    }else if(mode == "align_corners"){
        return TransformationMode::AlignCorners;
    }else if(mode == "asymmetric"){
        return TransformationMode::Asymmetric;
    }else if(mode == "tf_crop_and_resize"){
        return TransformationMode::TFCropAndResize;
    }else {  // constant
        return TransformationMode::HalfPixel;
    }
}

string getTransformationModeName(TransformationMode mode){
    if(mode == TransformationMode::HalfPixel){
        return "half_pixel";
    }else if(mode == TransformationMode::PytorchHalfPixel){
        return "pytorch_half_pixel";
    }else if(mode == TransformationMode::AlignCorners){
        return "align_corners";
    }else if(mode == TransformationMode::Asymmetric){
        return "asymmetric";
    }else if(mode == TransformationMode::TFCropAndResize){
        return "tf_crop_and_resize";
    }else {  // constant
        return "half_pixel";
    }
}

void show_deprecated_warning(const string& deprecated_name, const string& new_name, const string& type, const string& version){
    std::cerr << "[DEPRECATION WARNING]: The '" << deprecated_name << "' " << type << " will be deprecated in a " << version << " version";
    if (!new_name.empty()) { std::cerr << " in favor of '" << new_name << "'"; }
    std::cerr << "." << std::endl;
}


vector<string> read_lines_from_file(const string& filename){
    vector<string> lines;


    // Read file
    std::ifstream ifile(filename);

    // Check if the file exists
    if (!ifile) {
        throw std::runtime_error("The file does not exists. Filename: " + filename);
    }

    // Read lines
    std::string line;
    while (std::getline(ifile, line)){
        if(!line.empty()){  // Check if the line is empty
            lines.push_back(line);
        }
    }

    return lines;
}


// ---------------------------------------------------------------------------------------------
// Profiling

// profiling declarations
PROFILING_ENABLE(maximum);
PROFILING_ENABLE(minimum);
PROFILING_ENABLE(max);
PROFILING_ENABLE(argmax);
PROFILING_ENABLE(argmax_d);
PROFILING_ENABLE(min);
PROFILING_ENABLE(argmin);
PROFILING_ENABLE(sum);
PROFILING_ENABLE(sum_abs);
PROFILING_ENABLE(prod);
PROFILING_ENABLE(mean);
PROFILING_ENABLE(median);
PROFILING_ENABLE(std);
PROFILING_ENABLE(var);
PROFILING_ENABLE(mode);
PROFILING_ENABLE(abs);
PROFILING_ENABLE(acos);
PROFILING_ENABLE(add);
PROFILING_ENABLE(asin);
PROFILING_ENABLE(atan);
PROFILING_ENABLE(cell);
PROFILING_ENABLE(clamp);
PROFILING_ENABLE(clampmax);
PROFILING_ENABLE(clampmin);
PROFILING_ENABLE(cos);
PROFILING_ENABLE(cosh);
PROFILING_ENABLE(div);
PROFILING_ENABLE(exp);
PROFILING_ENABLE(floor);
PROFILING_ENABLE(inv);
PROFILING_ENABLE(log);
PROFILING_ENABLE(log2);
PROFILING_ENABLE(log10);
PROFILING_ENABLE(logn);
PROFILING_ENABLE(mod);
PROFILING_ENABLE(mult);
PROFILING_ENABLE(neg);
PROFILING_ENABLE(normalize);
PROFILING_ENABLE(pow);
PROFILING_ENABLE(powb);
PROFILING_ENABLE(reciprocal);
PROFILING_ENABLE(remainder);
PROFILING_ENABLE(round);
PROFILING_ENABLE(rsqrt);
PROFILING_ENABLE(sigmoid);
PROFILING_ENABLE(sign);
PROFILING_ENABLE(sin);
PROFILING_ENABLE(sinh);
PROFILING_ENABLE(sqr);
PROFILING_ENABLE(sqrt);
PROFILING_ENABLE(sub);
PROFILING_ENABLE(tan);
PROFILING_ENABLE(tanh);
PROFILING_ENABLE(trunc);
PROFILING_ENABLE(inc);
PROFILING_ENABLE(el_div);
PROFILING_ENABLE(mult2D);
PROFILING_ENABLE(el_mult);
PROFILING_ENABLE(sum2D_rowwise);
PROFILING_ENABLE(reduce_sum2D);
PROFILING_ENABLE(sum2D_colwise);
PROFILING_ENABLE(ceil);
// da
PROFILING_ENABLE(shift);
PROFILING_ENABLE(rotate);
PROFILING_ENABLE(scale);
PROFILING_ENABLE(flip);
PROFILING_ENABLE(crop);
PROFILING_ENABLE(crop_scale);
PROFILING_ENABLE(cutout);
PROFILING_ENABLE(shift_random);
PROFILING_ENABLE(rotate_random);
PROFILING_ENABLE(scale_random);
PROFILING_ENABLE(flip_random);
PROFILING_ENABLE(crop_random);
PROFILING_ENABLE(crop_scale_random);
PROFILING_ENABLE(cutout_random);
// reduction
PROFILING_ENABLE(reduce);
PROFILING_ENABLE(reduce_op);
PROFILING_ENABLE(reduction);
PROFILING_ENABLE(reduction_back);
// activations
PROFILING_ENABLE(ELu);
PROFILING_ENABLE(Exp);
PROFILING_ENABLE(ReLu);
PROFILING_ENABLE(Tanh);
PROFILING_ENABLE(D_ELu);
PROFILING_ENABLE(D_Exp);
PROFILING_ENABLE(D_Tanh);
PROFILING_ENABLE(D_ThresholdedReLu);
PROFILING_ENABLE(D_HardSigmoid);
PROFILING_ENABLE(D_LeakyRelu);
PROFILING_ENABLE(D_Linear);
PROFILING_ENABLE(D_ReLu);
PROFILING_ENABLE(D_LeakyReLu);
PROFILING_ENABLE(D_Sigmoid);
PROFILING_ENABLE(D_Softmax);
PROFILING_ENABLE(D_softplus);
PROFILING_ENABLE(HardSigmoid);
PROFILING_ENABLE(D_softsign);
PROFILING_ENABLE(LeakyReLu);
PROFILING_ENABLE(Linear);
PROFILING_ENABLE(Sigmoid);
PROFILING_ENABLE(Softmax);
PROFILING_ENABLE(Softplus);
PROFILING_ENABLE(Softsign);
PROFILING_ENABLE(ThresholdedReLu);
// conv
PROFILING_ENABLE(Conv2D);
PROFILING_ENABLE(Conv2DReLU);
PROFILING_ENABLE(Conv2D_grad);
PROFILING_ENABLE(Conv2D_back);
// losses
PROFILING_ENABLE(cent);
// generator
PROFILING_ENABLE(fill_rand_uniform);
PROFILING_ENABLE(fill_rand_signed_uniform);
PROFILING_ENABLE(fill_rand_normal);
PROFILING_ENABLE(fill_rand_binary);
// comparison
PROFILING_ENABLE(all);
PROFILING_ENABLE(any);
PROFILING_ENABLE(isfinite);
PROFILING_ENABLE(isinf);
PROFILING_ENABLE(isnan);
PROFILING_ENABLE(isneginf);
PROFILING_ENABLE(isposinf);
PROFILING_ENABLE(logical_and);
PROFILING_ENABLE(logical_or);
PROFILING_ENABLE(logical_not);
PROFILING_ENABLE(logical_xor);
PROFILING_ENABLE(allclose);
PROFILING_ENABLE(isclose);
PROFILING_ENABLE(greater);
PROFILING_ENABLE(greater_equal);
PROFILING_ENABLE(less);
PROFILING_ENABLE(less_equal);
PROFILING_ENABLE(equal);
PROFILING_ENABLE(not_equal);
PROFILING_ENABLE(equivalent);
// bn
PROFILING_ENABLE(permute_channels_last);
PROFILING_ENABLE(permute_channels_first);
PROFILING_ENABLE(permute_batch_last);
PROFILING_ENABLE(permute_batch_first);
// core_nn
PROFILING_ENABLE(repeat_nn);
PROFILING_ENABLE(d_repeat_nn);
PROFILING_ENABLE(select);
PROFILING_ENABLE(select_back);
PROFILING_ENABLE(set_select);
PROFILING_ENABLE(set_select_back);
PROFILING_ENABLE(transform);
// metrics
PROFILING_ENABLE(accuracy);
PROFILING_ENABLE(bin_accuracy);
// pool
PROFILING_ENABLE(MPool2D);
PROFILING_ENABLE(MPool2D_back);
PROFILING_ENABLE(AvgPool2D);
PROFILING_ENABLE(AvgPool2D_back);
// fpga-specific
PROFILING_ENABLE(fpga_hlsinf);
PROFILING_ENABLE(Precision_Conversion);
PROFILING_ENABLE(FPGA_READ);
PROFILING_ENABLE(FPGA_WRITE);

void __show_profile() {

  // profiling declarations
  PROFILING_PRINTF(maximum);
  PROFILING_PRINTF(minimum);
  PROFILING_PRINTF(max);
  PROFILING_PRINTF(argmax);
  PROFILING_PRINTF(argmax_d);
  PROFILING_PRINTF(min);
  PROFILING_PRINTF(argmin);
  PROFILING_PRINTF(sum);
  PROFILING_PRINTF(sum_abs);
  PROFILING_PRINTF(prod);
  PROFILING_PRINTF(mean);
  PROFILING_PRINTF(median);
  PROFILING_PRINTF(std);
  PROFILING_PRINTF(var);
  PROFILING_PRINTF(mode);
  PROFILING_PRINTF(abs);
  PROFILING_PRINTF(acos);
  PROFILING_PRINTF(add);
  PROFILING_PRINTF(asin);
  PROFILING_PRINTF(atan);
  PROFILING_PRINTF(cell);
  PROFILING_PRINTF(clamp);
  PROFILING_PRINTF(clampmax);
  PROFILING_PRINTF(clampmin);
  PROFILING_PRINTF(cos);
  PROFILING_PRINTF(cosh);
  PROFILING_PRINTF(div);
  PROFILING_PRINTF(exp);
  PROFILING_PRINTF(floor);
  PROFILING_PRINTF(inv);
  PROFILING_PRINTF(log);
  PROFILING_PRINTF(log2);
  PROFILING_PRINTF(log10);
  PROFILING_PRINTF(logn);
  PROFILING_PRINTF(mod);
  PROFILING_PRINTF(mult);
  PROFILING_PRINTF(neg);
  PROFILING_PRINTF(normalize);
  PROFILING_PRINTF(pow);
  PROFILING_PRINTF(powb);
  PROFILING_PRINTF(reciprocal);
  PROFILING_PRINTF(remainder);
  PROFILING_PRINTF(round);
  PROFILING_PRINTF(rsqrt);
  PROFILING_PRINTF(sigmoid);
  PROFILING_PRINTF(sign);
  PROFILING_PRINTF(sin);
  PROFILING_PRINTF(sinh);
  PROFILING_PRINTF(sqr);
  PROFILING_PRINTF(sqrt);
  PROFILING_PRINTF(sub);
  PROFILING_PRINTF(tan);
  PROFILING_PRINTF(tanh);
  PROFILING_PRINTF(trunc);
  PROFILING_PRINTF(inc);
  PROFILING_PRINTF(el_div);
  PROFILING_PRINTF(mult2D);
  PROFILING_PRINTF(el_mult);
  PROFILING_PRINTF(sum2D_rowwise);
  PROFILING_PRINTF(reduce_sum2D);
  PROFILING_PRINTF(sum2D_colwise);
  PROFILING_PRINTF(ceil);
  // da
  PROFILING_PRINTF(shift);
  PROFILING_PRINTF(rotate);
  PROFILING_PRINTF(scale);
  PROFILING_PRINTF(flip);
  PROFILING_PRINTF(crop);
  PROFILING_PRINTF(crop_scale);
  PROFILING_PRINTF(cutout);
  PROFILING_PRINTF(shift_random);
  PROFILING_PRINTF(rotate_random);
  PROFILING_PRINTF(scale_random);
  PROFILING_PRINTF(flip_random);
  PROFILING_PRINTF(crop_random);
  PROFILING_PRINTF(crop_scale_random);
  PROFILING_PRINTF(cutout_random);
  //reduction
  PROFILING_PRINTF(reduce);
  PROFILING_PRINTF(reduce_op);
  PROFILING_PRINTF(reduction);
  PROFILING_PRINTF(reduction_back);
  // activations
  PROFILING_PRINTF(ELu);
  PROFILING_PRINTF(Exp);
  PROFILING_PRINTF(ReLu);
  PROFILING_PRINTF(Tanh);
  PROFILING_PRINTF(D_ELu);
  PROFILING_PRINTF(D_Exp);
  PROFILING_PRINTF(D_Tanh);
  PROFILING_PRINTF(D_ThresholdedReLu);
  PROFILING_PRINTF(D_HardSigmoid);
  PROFILING_PRINTF(D_LeakyRelu);
  PROFILING_PRINTF(D_Linear);
  PROFILING_PRINTF(D_ReLu);
  PROFILING_PRINTF(D_LeakyReLu);
  PROFILING_PRINTF(D_Sigmoid);
  PROFILING_PRINTF(D_Softmax);
  PROFILING_PRINTF(D_softplus);
  PROFILING_PRINTF(HardSigmoid);
  PROFILING_PRINTF(D_softsign);
  PROFILING_PRINTF(LeakyReLu);
  PROFILING_PRINTF(Linear);
  PROFILING_PRINTF(Sigmoid);
  PROFILING_PRINTF(Softmax);
  PROFILING_PRINTF(Softplus);
  PROFILING_PRINTF(Softsign);
  PROFILING_PRINTF(ThresholdedReLu);
  // conv
  PROFILING_PRINTF(Conv2D);
  PROFILING_PRINTF(Conv2DReLU);
  PROFILING_PRINTF(Conv2D_grad);
  PROFILING_PRINTF(Conv2D_back);
  // losses
  PROFILING_PRINTF(cent);
  // generator
  PROFILING_PRINTF(fill_rand_uniform);
  PROFILING_PRINTF(fill_rand_signed_uniform);
  PROFILING_PRINTF(fill_rand_normal);
  PROFILING_PRINTF(fill_rand_binary);
  // comparison
  PROFILING_PRINTF(all);
  PROFILING_PRINTF(any);
  PROFILING_PRINTF(isfinite);
  PROFILING_PRINTF(isinf);
  PROFILING_PRINTF(isnan);
  PROFILING_PRINTF(isneginf);
  PROFILING_PRINTF(isposinf);
  PROFILING_PRINTF(logical_and);
  PROFILING_PRINTF(logical_or);
  PROFILING_PRINTF(logical_not);
  PROFILING_PRINTF(logical_xor);
  PROFILING_PRINTF(allclose);
  PROFILING_PRINTF(isclose);
  PROFILING_PRINTF(greater);
  PROFILING_PRINTF(greater_equal);
  PROFILING_PRINTF(less);
  PROFILING_PRINTF(less_equal);
  PROFILING_PRINTF(equal);
  PROFILING_PRINTF(not_equal);
  PROFILING_PRINTF(equivalent);
  // bn
  PROFILING_PRINTF(permute_channels_last);
  PROFILING_PRINTF(permute_channels_first);
  PROFILING_PRINTF(permute_batch_last);
  PROFILING_PRINTF(permute_batch_first);
  // core_nn
  PROFILING_PRINTF(repeat_nn);
  PROFILING_PRINTF(d_repeat_nn);
  PROFILING_PRINTF(select);
  PROFILING_PRINTF(select_back);
  PROFILING_PRINTF(set_select);
  PROFILING_PRINTF(set_select_back);
  PROFILING_PRINTF(transform);
  // metrics
  PROFILING_PRINTF(accuracy);
  PROFILING_PRINTF(bin_accuracy);
  // pool
  PROFILING_PRINTF(MPool2D);
  PROFILING_PRINTF(MPool2D_back);
  PROFILING_PRINTF(AvgPool2D);

  // fpga-specific
  PROFILING_PRINTF(fpga_hlsinf);
  PROFILING_PRINTF(Precision_Conversion);
  PROFILING_PRINTF(FPGA_READ);
  PROFILING_PRINTF(FPGA_WRITE);

}

void __reset_profile() {

  // profiling declarations
  PROFILING_RESET(maximum);
  PROFILING_RESET(minimum);
  PROFILING_RESET(max);
  PROFILING_RESET(argmax);
  PROFILING_RESET(argmax_d);
  PROFILING_RESET(min);
  PROFILING_RESET(argmin);
  PROFILING_RESET(sum);
  PROFILING_RESET(sum_abs);
  PROFILING_RESET(prod);
  PROFILING_RESET(mean);
  PROFILING_RESET(median);
  PROFILING_RESET(std);
  PROFILING_RESET(var);
  PROFILING_RESET(mode);
  PROFILING_RESET(abs);
  PROFILING_RESET(acos);
  PROFILING_RESET(add);
  PROFILING_RESET(asin);
  PROFILING_RESET(atan);
  PROFILING_RESET(cell);
  PROFILING_RESET(clamp);
  PROFILING_RESET(clampmax);
  PROFILING_RESET(clampmin);
  PROFILING_RESET(cos);
  PROFILING_RESET(cosh);
  PROFILING_RESET(div);
  PROFILING_RESET(exp);
  PROFILING_RESET(floor);
  PROFILING_RESET(inv);
  PROFILING_RESET(log);
  PROFILING_RESET(log2);
  PROFILING_RESET(log10);
  PROFILING_RESET(logn);
  PROFILING_RESET(mod);
  PROFILING_RESET(mult);
  PROFILING_RESET(neg);
  PROFILING_RESET(normalize);
  PROFILING_RESET(pow);
  PROFILING_RESET(powb);
  PROFILING_RESET(reciprocal);
  PROFILING_RESET(remainder);
  PROFILING_RESET(round);
  PROFILING_RESET(rsqrt);
  PROFILING_RESET(sigmoid);
  PROFILING_RESET(sign);
  PROFILING_RESET(sin);
  PROFILING_RESET(sinh);
  PROFILING_RESET(sqr);
  PROFILING_RESET(sqrt);
  PROFILING_RESET(sub);
  PROFILING_RESET(tan);
  PROFILING_RESET(tanh);
  PROFILING_RESET(trunc);
  PROFILING_RESET(inc);
  PROFILING_RESET(el_div);
  PROFILING_RESET(mult2D);
  PROFILING_RESET(el_mult);
  PROFILING_RESET(sum2D_rowwise);
  PROFILING_RESET(reduce_sum2D);
  PROFILING_RESET(sum2D_colwise);
  PROFILING_RESET(ceil);
  // da
  PROFILING_RESET(shift);
  PROFILING_RESET(rotate);
  PROFILING_RESET(scale);
  PROFILING_RESET(flip);
  PROFILING_RESET(crop);
  PROFILING_RESET(crop_scale);
  PROFILING_RESET(cutout);
  PROFILING_RESET(shift_random);
  PROFILING_RESET(rotate_random);
  PROFILING_RESET(scale_random);
  PROFILING_RESET(flip_random);
  PROFILING_RESET(crop_random);
  PROFILING_RESET(crop_scale_random);
  PROFILING_RESET(cutout_random);
  //reduction
  PROFILING_RESET(reduce);
  PROFILING_RESET(reduce_op);
  PROFILING_RESET(reduction);
  PROFILING_RESET(reduction_back);
  // activations
  PROFILING_RESET(ELu);
  PROFILING_RESET(Exp);
  PROFILING_RESET(ReLu);
  PROFILING_RESET(Tanh);
  PROFILING_RESET(D_ELu);
  PROFILING_RESET(D_Exp);
  PROFILING_RESET(D_Tanh);
  PROFILING_RESET(D_ThresholdedReLu);
  PROFILING_RESET(D_HardSigmoid);
  PROFILING_RESET(D_LeakyRelu);
  PROFILING_RESET(D_Linear);
  PROFILING_RESET(D_ReLu);
  PROFILING_RESET(D_LeakyReLu);
  PROFILING_RESET(D_Sigmoid);
  PROFILING_RESET(D_Softmax);
  PROFILING_RESET(D_softplus);
  PROFILING_RESET(HardSigmoid);
  PROFILING_RESET(D_softsign);
  PROFILING_RESET(LeakyReLu);
  PROFILING_RESET(Linear);
  PROFILING_RESET(Sigmoid);
  PROFILING_RESET(Softmax);
  PROFILING_RESET(Softplus);
  PROFILING_RESET(Softsign);
  PROFILING_RESET(ThresholdedReLu);
  // conv
  PROFILING_RESET(Conv2D);
  PROFILING_RESET(Conv2DReLU);
  PROFILING_RESET(Conv2D_grad);
  PROFILING_RESET(Conv2D_back);
  // losses
  PROFILING_RESET(cent);
  // generator
  PROFILING_RESET(fill_rand_uniform);
  PROFILING_RESET(fill_rand_signed_uniform);
  PROFILING_RESET(fill_rand_normal);
  PROFILING_RESET(fill_rand_binary);
  // comparison
  PROFILING_RESET(all);
  PROFILING_RESET(any);
  PROFILING_RESET(isfinite);
  PROFILING_RESET(isinf);
  PROFILING_RESET(isnan);
  PROFILING_RESET(isneginf);
  PROFILING_RESET(isposinf);
  PROFILING_RESET(logical_and);
  PROFILING_RESET(logical_or);
  PROFILING_RESET(logical_not);
  PROFILING_RESET(logical_xor);
  PROFILING_RESET(allclose);
  PROFILING_RESET(isclose);
  PROFILING_RESET(greater);
  PROFILING_RESET(greater_equal);
  PROFILING_RESET(less);
  PROFILING_RESET(less_equal);
  PROFILING_RESET(equal);
  PROFILING_RESET(not_equal);
  PROFILING_RESET(equivalent);
  // bn
  PROFILING_RESET(permute_channels_last);
  PROFILING_RESET(permute_channels_first);
  PROFILING_RESET(permute_batch_last);
  PROFILING_RESET(permute_batch_first);
  // core_nn
  PROFILING_RESET(repeat_nn);
  PROFILING_RESET(d_repeat_nn);
  PROFILING_RESET(select);
  PROFILING_RESET(select_back);
  PROFILING_RESET(set_select);
  PROFILING_RESET(set_select_back);
  PROFILING_RESET(transform);
  // metrics
  PROFILING_RESET(accuracy);
  PROFILING_RESET(bin_accuracy);
  // pool
  PROFILING_RESET(MPool2D);
  PROFILING_RESET(MPool2D_back);
  PROFILING_RESET(AvgPool2D);

  // fpga-specific
  PROFILING_RESET(fpga_hlsinf);
  PROFILING_RESET(Precision_Conversion);
  PROFILING_RESET(FPGA_READ);
  PROFILING_RESET(FPGA_WRITE);
}

