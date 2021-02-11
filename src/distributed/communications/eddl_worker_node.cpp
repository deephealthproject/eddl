/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#include <eddl/distributed/eddl_worker_node.h>

#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>


namespace eddl {

eddl_worker_node::eddl_worker_node(std::string description)
{
// ip:192.168.13.11;cpu:2,8192;gpu:1,low_mem;fpga:0,0;batch_size:10;
    std::vector<std::string> columns=str_split(description, ';');

    for (auto s : columns) {
        std::vector<std::string> key_values=str_split(s, ':');
        std::string key = key_values[0];
        std::vector<std::string> values=str_split(key_values[1], ',');

        if (key == "ip") {

            this->hostname_or_ip_address = values[0];
            struct hostent *host = gethostbyname(this->hostname_or_ip_address.c_str());
            if (sizeof(this->s_addr) != host->h_length)
                throw std::runtime_error(err_msg("address error conversion."));
            memcpy((char *)&this->s_addr, (char *)host->h_addr, host->h_length);

        } else if (key == "cpu") {

            this->cpu_cores = std::stoi(values[0]);
            this->cpu_mem = std::stoi(values[1]);

        } else if (key == "gpu") {

            this->gpu_cards = std::stoi(values[0]);
            this->gpu_mem_mode = values[1];

        } else if (key == "fga") {

            this->fpga_cards = std::stoi(values[0]);
            this->fpga_mem = std::stoi(values[1]);

        } else if (key == "batch_size") {

            this->batch_size = std::stoi(values[0]);
        }
    }

    this->data_subset="";
    this->active = true;
}

std::string eddl_worker_node::get_ip_address()
{
    return eddl::get_ip_address(this->s_addr);
}


};
