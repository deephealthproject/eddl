/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#include <map>
#include <chrono>

#include <eddl/distributed/eddl_distributed.h>

#include <iostream>
#include <sstream>

namespace eddl {

uint64_t get_system_milliseconds()
{
    std::chrono::system_clock::time_point   just_now = std::chrono::system_clock::now();
    std::chrono::system_clock::duration     since_epoch = just_now.time_since_epoch();
    uint64_t msec = std::chrono::duration_cast<std::chrono::milliseconds>(since_epoch).count();

    return msec;
}
size_t compute_aligned_size(size_t size)
{
    return size + ((eddl_alignment - (size % eddl_alignment)) % eddl_alignment);
}

void * eddl_malloc(size_t size)
{
    // the memory allocated by this method must be released by using the free() system call
    void *ptr = nullptr;
    int rc = posix_memalign(&ptr, eddl_alignment, compute_aligned_size(size));
    if (rc != 0)
        throw std::runtime_error(err_msg("error allocating memory."));

    return ptr;
}

std::vector<std::string> str_split(std::string s, char sep)
{
    std::vector<std::string>    v;

    std::string part="";

    size_t i = 0;
    while (i < s.size()) {

        size_t j = s.find(sep, i);

        if (j==i) {
            v.push_back(std::string(""));
            i+=1;
        } else if (j > s.size()) { // end of string reached
            v.push_back(std::string(s.substr(i,s.size()-i)));
            i=s.size();
        } else {
            v.push_back(std::string(s.substr(i,j-i)));
            i=j+1;
        }
    }

    return v;
}

std::string get_ip_address(uint32_t s_addr)
{
    unsigned char *ptr = (unsigned char *) &s_addr;
    unsigned int   i;

    std::string s1, s2, s3, s4;

    i = ptr[0]; s1 = std::to_string(i);
    i = ptr[1]; s2 = std::to_string(i);
    i = ptr[2]; s3 = std::to_string(i);
    i = ptr[3]; s4 = std::to_string(i);

    return s1 + "." + s2 + "." + s3 + "." + s4;
}
std::string pointer_to_string(void * ptr)
{
    std::stringstream buff;
    buff << ptr;
    return buff.str();
}

std::string compose_log_message(const char * filename, const int line_number, const char * function_name, const char * msg)
{
    std::stringstream buff;
    buff << filename;
    buff << ":";
    buff << std::to_string(line_number);
    buff << ":";
    buff << function_name;
    buff << ": ";
    buff << msg;
    return buff.str();
}
std::string compose_log_message(const char * filename, const int line_number, const char * function_name, std::string msg)
{
    std::stringstream buff;
    buff << filename << ":" << std::to_string(line_number) + ":" << function_name << ": " << msg;
    return buff.str();
}
void print_log_message(const char * filename, const int line_number, const char * function_name, const char * msg)
{
    std::cout << compose_log_message(filename, line_number, function_name, msg) << std::endl;
}
void print_log_message(const char * filename, const int line_number, const char * function_name, std::string msg)
{
    std::cout << compose_log_message(filename, line_number, function_name, msg) << std::endl;
}
void print_err_message(const char * filename, const int line_number, const char * function_name, const char * msg)
{
    std::cerr << compose_log_message(filename, line_number, function_name, msg) << std::endl;
}
void print_err_message(const char * filename, const int line_number, const char * function_name, std::string msg)
{
    std::cerr << compose_log_message(filename, line_number, function_name, msg) << std::endl;
}


static std::map<int, std::string>  __eddl_message_types_names;
void init_message_type_names()
{
    #define stringify(name) # name
    __eddl_message_types_names[DATA_SAMPLES] = stringify(DATA_SAMPLES);
    __eddl_message_types_names[DATA_WEIGHTS] = stringify(DATA_WEIGHTS);
    __eddl_message_types_names[DATA_GRADIENTS] = stringify(DATA_GRADIENTS);
    __eddl_message_types_names[MSG_ACK_SAMPLES] = stringify(MSG_ACK_SAMPLES);
    __eddl_message_types_names[MSG_ACK_WEIGHTS] = stringify(MSG_ACK_WEIGHTS);
    __eddl_message_types_names[MSG_ACK_GRADIENTS] = stringify(MSG_ACK_GRADIENTS);
    __eddl_message_types_names[PARAMETER] = stringify(PARAMETER);
    __eddl_message_types_names[COMMAND] = stringify(COMMAND);
    __eddl_message_types_names[PKG_ACK] = stringify(PKG_ACK);
    __eddl_message_types_names[MSG_CHKSUM] = stringify(MSG_CHKSUM);
    #undef stringify
}
std::string get_message_type_name(int value)
{
    return __eddl_message_types_names[value];
}
void show_all_message_type_names()
{
    for(auto iter: __eddl_message_types_names)
        std::cout << std::hex << iter.first << " " << iter.second << std::endl;
}


};
