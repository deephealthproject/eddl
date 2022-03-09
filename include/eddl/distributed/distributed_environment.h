/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: August 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#ifndef __DISTRIBUTED_ENVIRONMENT_H__
#define __DISTRIBUTED_ENVIRONMENT_H__ 1

#include <eddl/distributed/eddl_distributed.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

namespace eddl {

class DistributedEnvironment
{
public:
    DistributedEnvironment()
    {
        this->master_ip_addr = "";
        this->master_s_addr = 0;
        this->tcp_port      = base_tcp_port;
        this->udp_data_port = base_udp_data_port;
        this->udp_ack_port  = base_udp_ack_port;
        this->my_ip_addr = "";
        this->my_s_addr = 0;
        this->set_multicast_group_addr(eddl_multicast_group_addr);
        this->verbose_level = 0;
        this->batch_size = 10;

        init_message_type_names();
    }

    bool is_master_ip_addr_set()
    {
        return this->master_s_addr != 0;
    }
    std::string get_master_ip_addr()
    {
        if (this->master_s_addr == 0)
            throw std::runtime_error(err_msg("master_ip_addr is not set yet!"));

        return this->master_ip_addr;
    }
    std::string get_my_ip_addr()
    {
        if (this->my_s_addr == 0)
            throw std::runtime_error(err_msg("my_ip_addr not is not set yet!"));

        return this->my_ip_addr;
    }
    std::string get_multicast_group_addr()
    {
        if (this->multicast_s_addr == 0)
            throw std::runtime_error(err_msg("multicast_group_addr not is not set yet!"));

        return this->multicast_group_addr;
    }
    void set_master_ip_addr(std::string s)
    {
        struct in_addr addr;
        int rc = inet_aton(s.c_str(), &addr);
        if (rc == 1) {
            this->master_ip_addr = s;
            this->master_s_addr = addr.s_addr;
        } else {
            throw std::runtime_error(err_msg("invalid ip addr provided: " + s));
        }
    }
    void set_master_s_addr(in_addr_t s_addr)
    {
        if (this->master_s_addr != 0) {
            throw std::runtime_error(err_msg("master_ip_addr is set, clear it before setting a new one!"));
        }

        struct in_addr addr;
        addr.s_addr = s_addr;
        char * s = inet_ntoa(addr);

        this->master_ip_addr = s;
        this->master_s_addr = s_addr;
    }
    void clear_master_ip_addr()
    {
        this->master_ip_addr = "";
        this->master_s_addr = 0;
    }
    void set_my_ip_addr(std::string s)
    {
        struct in_addr addr;
        int rc = inet_aton(s.c_str(), &addr);
        if (rc == 1) {
            this->my_ip_addr = s;
            this->my_s_addr = addr.s_addr;
        } else {
            throw std::runtime_error(err_msg("invalid ip addr provided: " + s));
        }
    }
    void set_multicast_group_addr(std::string s)
    {
        struct in_addr addr;
        int rc = inet_aton(s.c_str(), &addr);
        if (rc == 1) {
            this->multicast_group_addr = s;
            this->multicast_s_addr = addr.s_addr;
        } else {
            throw std::runtime_error(err_msg("invalid ip addr provided: " + s));
        }
    }
    in_addr_t get_master_s_addr()
    {
        if (this->master_s_addr == 0)
            throw std::runtime_error(err_msg("master_ip_addr is not set yet!"));

         return this->master_s_addr;
    }
    in_addr_t get_my_s_addr()
    {
        if (this->my_s_addr == 0)
            throw std::runtime_error(err_msg("my_ip_addr is not set yet!"));

         return this->my_s_addr;
    }
    in_addr_t get_multicast_s_addr()
    {
        if (this->multicast_s_addr == 0)
            throw std::runtime_error(err_msg("multicast_group_addr is not set yet!"));

         return this->multicast_s_addr;
    }

    int get_verbose_level() { return this->verbose_level; }
    void set_verbose_level(int v) { this->verbose_level = std::max(0,v); }
    void increase_verbose_level() { this->verbose_level++; }

    int get_batch_size() { return this->batch_size; }
    void set_batch_size(int bs) { this->batch_size = std::max(1,bs); }

    int get_tcp_port() { return this->tcp_port; }
    void set_tcp_port(int port_number) { this->tcp_port = port_number; }

    int get_udp_data_port() { return this->udp_data_port; }
    void set_udp_data_port(int port_number) { this->udp_data_port = port_number; }

    int get_udp_ack_port() { return this->udp_ack_port; }
    void set_udp_ack_port(int port_number) { this->udp_ack_port = port_number; }


private:
    std::string     master_ip_addr;
    in_addr_t       master_s_addr;
    int             tcp_port;
    int             udp_data_port;
    int             udp_ack_port;
    std::string     my_ip_addr;
    in_addr_t       my_s_addr;
    int             verbose_level;
    int             batch_size;
    std::string     multicast_group_addr;
    in_addr_t       multicast_s_addr;

}; // of class DistributedEnvironment
}; // of namespace eddl

#endif // __DISTRIBUTED_ENVIRONMENT_H__
