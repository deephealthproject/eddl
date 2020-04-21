/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
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
#include <math.h>
#include <vector>
#include <iomanip>
#include <limits>

#include "eddl/system_info.h"
#include "eddl/utils.h"

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


void msg(const string& text, const string& title) {
    string s(text);
    if(!title.empty()){
        s += " (" + title + ")";
    }
    throw std::runtime_error(s);
}


float *get_fmem(long int size, const string &str){
    // Careful with memory overcommitment:
    // https://stackoverflow.com/questions/48585079/malloc-on-linux-without-overcommitting
    // TODO: This function does not work properly (...but it does, at least most of the time -for linux and mac-)
    float* ptr = nullptr;
    bool error = false;


    // Check if free memory is bigger than requested
    unsigned long freemem = get_free_mem();
    if (size*sizeof(float) > freemem) {
        error=true;
    }

    // New vs Malloc *******************
    // New is the C++ way of doing it
    // New is type-safe, Malloc is not
    // New calls your type constructor, Malloc not - Same for destructor
    // New is an operator, Malloc a function (slower)
    try{
        ptr = new float[size];
    }
    catch (std::bad_alloc& badAlloc){
        error=true;
    }

    // Check for errors
    // Not enough free memory
    if (error) {
        delete ptr;
        throw std::runtime_error("Error allocating " + string(bytes2human(size * sizeof(float))) + " in " + string(str));
    }

    return ptr;
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
        std::string type = "MemFree:";
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
                max = shape[i]-1;
            }else{
                if (pos==0){ // ":5"
                    min = 0;
                    max = std::stoi(str.substr(pos+delimiter.length(), shape[i]-1)) - 1;
                }else if(pos==str.length()-1){  // "5:"
                    min = std::stoi(str.substr(0, pos));
                    max = shape[i]-1;
                }else{  // "5:10"
                    min = std::stoi(str.substr(0, pos));
                    max = std::stoi(str.substr(pos+delimiter.length(), shape[i]-1)) - 1;
                }
            }
        }else{  // Not found => "5"
            min = std::stoi(str);
            max = min;
        }

        // Negative indices // len + (-x)
        if(min<0) { min = shape[i] + min ; }
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

    return addresses;
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

    return addresses;
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