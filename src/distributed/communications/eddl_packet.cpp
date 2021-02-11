/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#include <eddl/distributed/eddl_packet.h>

#include <cstring>
#include <openssl/sha.h>
#include <iostream>

namespace eddl {

eddl_packet::eddl_packet(uint32_t type,
                         uint32_t source_addr,
                         uint32_t target_addr,
                         std::string & message_id,
                         size_t seq_no,
                         size_t seq_len,
                         size_t message_size,
                         uint32_t all_but_last_packet_size,
                         size_t data_size,
                         void * data )
{
    if (data_size == 0)
        throw std::runtime_error(err_msg("packet data size cannot be zero."));

    if (data_size > eddl_packet_data_size)
        throw std::runtime_error(err_msg("packet data size cannot be larger than 'eddl_packet_data_size'."));

    if (eddl_msg_id_len != message_id.size())
        throw std::runtime_error(err_msg("non-valid message id."));

    this->type = type;
    this->source_addr = source_addr;
    this->target_addr = target_addr;
    memset(this->message_id, 0, sizeof(this->message_id));
    strncpy(this->message_id, message_id.c_str(), eddl_msg_id_len);
    this->seq_no = seq_no;
    this->seq_len = seq_len;
    this->message_size = message_size;
    this->all_but_last_packet_size = all_but_last_packet_size;
    this->data_size = data_size;
    memset(this->data, 0, sizeof(this->data));
    memcpy(this->data, data, data_size);
    this->compute_checksum();
}
eddl_packet::eddl_packet(uint32_t type,
                         uint32_t source_addr,
                         uint32_t target_addr,
                         std::string & message_id,
                         size_t seq_no,
                         size_t seq_len,
                         uint32_t command )
{
    if (eddl_msg_id_len != message_id.size())
        throw std::runtime_error(err_msg("non-valid message id."));

    this->type = type;
    this->source_addr = source_addr;
    this->target_addr = target_addr;
    memset(this->message_id, 0, sizeof(this->message_id));
    strncpy(this->message_id, message_id.c_str(), eddl_msg_id_len);
    this->seq_no = seq_no;
    this->seq_len = seq_len;
    this->message_size = 0;
    this->all_but_last_packet_size = 0; // TO-BE REVIEWED
    this->data_size = sizeof(uint32_t);
    memset(this->data, 0, sizeof(this->data));
    uint32_t *p = (uint32_t *)this->data;
    *p = command;
    this->compute_checksum();
}

eddl_packet::~eddl_packet()
{
}

uint32_t eddl_packet::get_command()
{
    uint32_t *p = (uint32_t *)this->data;

    return *p;
}

void eddl_packet::compute_checksum()
{
    memset(this->checksum, 0, eddl_checksum_len*sizeof(unsigned char));
    unsigned char checksum[eddl_checksum_len];
    SHA256((unsigned char *)this, sizeof(eddl_packet), checksum);
    memcpy(this->checksum, checksum, eddl_checksum_len*sizeof(unsigned char));
}
bool eddl_packet::is_checksum_valid()
{
    unsigned char checksum_orig[eddl_checksum_len];
    unsigned char checksum_new[eddl_checksum_len];
    memcpy(checksum_orig, this->checksum, eddl_checksum_len*sizeof(unsigned char));
    memset(this->checksum, 0, eddl_checksum_len*sizeof(unsigned char));
    SHA256((unsigned char *)this, sizeof(eddl_packet), checksum_new);
    memcpy(this->checksum, checksum_orig, eddl_checksum_len*sizeof(unsigned char));

    for (int i=0; i < eddl_checksum_len; i++)
        if (checksum_orig[i] != checksum_new[i]) return false;

    return true;
}

eddl_packet_ack * eddl_packet::create_acknowledgement(uint32_t worker_addr)
{
    return new eddl_packet_ack(worker_addr, this->seq_no, this->message_id);
}

};
