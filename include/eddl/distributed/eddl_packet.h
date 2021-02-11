/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#ifndef __EDDL_PACKET_H__
#define __EDDL_PACKET_H__ 1

#include <string>

#include <eddl/distributed/eddl_distributed.h>
#include <eddl/distributed/eddl_packet_ack.h>

namespace eddl {

class eddl_packet
{
public:
    eddl_packet(uint32_t type,
                uint32_t source_addr,
                uint32_t target_addr,
                std::string  & message_id,
                size_t seq_no,
                size_t seq_len,
                size_t message_size,
                uint32_t all_but_last_packet_size,
                size_t data_size,
                void * data);
    eddl_packet(uint32_t type,
                uint32_t source_addr,
                uint32_t target_addr,
                std::string &  message_id,
                size_t seq_no,
                size_t seq_len,
                uint32_t command);
    ~eddl_packet();

    inline uint32_t get_type() { return type; }
    inline uint32_t get_source_addr() { return source_addr; }
    inline uint32_t get_target_addr() { return target_addr; }
    inline size_t get_seq_no() { return seq_no; }
    inline size_t get_seq_len() { return seq_len; }
    inline void * get_data() { return data; }

    uint32_t get_command();

    inline unsigned char * get_checksum_ptr() { return checksum; }
    inline std::string get_checksum() { return std::string((char *)checksum, eddl_checksum_len); }
    std::string get_ip_address();

    std::string get_message_id() { return std::string(message_id, eddl_msg_id_len); }
    inline char * get_message_id_ptr() { return message_id; }
    // returns the size in bytes of the whole message this packet belongs to
    inline size_t get_message_size() { return message_size; }
    // return the size in bytes of the data contained in this packet
    inline size_t get_data_size() { return data_size; }
    // return the size in bytes of the all the packets of the same message but the last one
    inline size_t get_all_but_last_packet_size() { return all_but_last_packet_size; }

    void compute_checksum();
    bool is_checksum_valid();

    eddl_packet_ack * create_acknowledgement(uint32_t worker_addr);

private:
    uint32_t        type;
    uint32_t        source_addr;
    uint32_t        target_addr;
    uint32_t        all_but_last_packet_size;
    char            message_id[_eddl_msg_id_len_];
    size_t          seq_no;
    size_t          seq_len;
    size_t          message_size;
    size_t          data_size; // must be less than or equal to eddl_packet_data_size
    unsigned char   checksum[eddl_checksum_len];
    unsigned char   data[eddl_packet_data_size];
};

};

#endif // __EDDL_PACKET_H__
