/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#ifndef __EDDL_WORKER_NODE_H__
#define __EDDL_WORKER_NODE_H__ 1

#include <ctime>
#include <string>
#include <queue>

#include <eddl/distributed/eddl_distributed.h>

namespace eddl {

class eddl_worker_node
{
public:
    eddl_worker_node(std::string description);

    inline int get_cpu_cores() { return cpu_cores; }
    inline int get_cpu_mem() { return cpu_cores; }
    inline int get_gpu_cards() { return gpu_cards; }
    inline std::string get_gpu_mem() { return gpu_mem_mode; }
    inline int get_fpga_cards() { return fpga_cards; }
    inline int get_fpga_mem() { return fpga_mem; }
    inline int get_batch_size() { return batch_size; }

    inline void set_batch_size(int b) { batch_size=b; }

    std::string get_ip_address();
    inline uint32_t get_s_addr() { return s_addr; }

    inline bool is_active() { return active; }
    void activate() { active=true; }
    void deactivate() { active=false; }


private:
    std::string hostname_or_ip_address;
    uint32_t    s_addr;
    int         cpu_cores;
    int         cpu_mem; // in megas
    int         gpu_cards;
    std::string gpu_mem_mode; // "low_mem", "mid_mem", "full_mem"
    int         fpga_cards;
    int         fpga_mem; // in megas
    int         batch_size;
    bool        active;

    std::string data_subset; // short description or identifier of the subset assigned to the worker node

    std::queue<time_t>  gradient_timestamps;
};

};

/*
 *  computing service example

    ip:192.168.13.11;cpu:2,8192;gpu:1,low_mem;fpga:0,0;batch_size:10;

    this line describes a worker node whose ip address is 192.168.13.11,
    from which this task will use 2 cores assuming 8 GB is the total RAM of
    the computer, one GPU in low_mem mode will be used, 0 FPGAs are available
    with 0 MB of memory, and the batch_size used in the work node by the
    train_batch() method will be of 10 samples.
 */

#endif // __EDDL_WORKER_NODE_H__
