/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: August 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#ifndef __EDDL_PACKET_ACK_H__
#define __EDDL_PACKET_ACK_H__ 1

#include <cstring>
#include <string>

#include <eddl/distributed/eddl_distributed.h>

namespace eddl {

class eddl_packet_ack
{
public:
    eddl_packet_ack(uint32_t source_addr,
                    uint32_t seq_no,
                    char * message_id) :
        source_addr(source_addr),
        seq_no(seq_no)
    {
        memset(this->message_id, 0, sizeof(this->message_id));
        strncpy(this->message_id, message_id, eddl_msg_id_len);
    }

    ~eddl_packet_ack()
    {
    }

    inline uint32_t get_source_addr() { return this->source_addr; }
    inline size_t get_seq_no() { return this->seq_no; }
    std::string get_message_id() { return std::string(this->message_id, eddl_msg_id_len); }
    inline char * get_message_id_ptr() { return this->message_id; }

private:
    uint32_t        source_addr;
    uint32_t        seq_no;
    char            message_id[_eddl_msg_id_len_];
};
};
#endif // __EDDL_PACKET_ACK_H__
