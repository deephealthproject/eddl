/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#include <eddl/distributed/eddl_message.h>

#include <cstring>
#include <openssl/sha.h>
#include <iostream>
#include <iomanip>

namespace eddl {

eddl_message::eddl_message(uint32_t type,
                           uint32_t source_addr,
                           uint32_t target_addr,
                           size_t message_data_size,
                           size_t packet_data_size,
                           void * data )
: type(type), source_addr(source_addr), target_addr(target_addr)
{
    this->timestamp = get_system_milliseconds();
    this->set_message_id();
    this->message_data_size = 0;
    this->packet_data_size = 0;
    this->seq_len = 0;
    this->data = nullptr;

    this->checksum_has_been_set = false;
    memset(this->checksum, 0, eddl_checksum_len);

    this->received_packet = nullptr;

    if (0 == message_data_size && nullptr != data)
        throw std::runtime_error(err_msg("data size equal to zero when data is not nullptr."));

    if (type == eddl_message_types::COMMAND) {
        this->data = (unsigned char *)eddl_malloc(sizeof(uint32_t));
        uint32_t *p = (uint32_t *)this->data;
        // see fabric constructors in eddl_message.h
        // the command id is passed via the parameter 'packet_data_size'
        *p = (uint32_t)packet_data_size;
        this->message_data_size = this->packet_data_size = sizeof(uint32_t);
        this->seq_len = 1;
        this->compute_checksum();
        this->checksum_has_been_set = true;
    } else if (message_data_size > 0) {
        this->message_data_size = message_data_size;
        this->packet_data_size = packet_data_size;
        set_data(message_data_size, data);
    }
}
eddl_message::eddl_message(eddl_packet * packet)
{
    this->type = packet->get_type();
    this->source_addr = packet->get_source_addr();
    this->target_addr = packet->get_target_addr();
    this->timestamp = get_system_milliseconds();
    this->seq_len = 1;
    this->message_data_size = packet->get_data_size();
    this->packet_data_size = packet->get_data_size();
    this->set_message_id(packet->get_message_id_ptr());
    memset(this->checksum, 0, eddl_checksum_len*sizeof(unsigned char));
    this->data = (unsigned char *)eddl_malloc(this->message_data_size);
    memcpy(this->data, packet->get_data(), this->message_data_size);
    this->pending_packets = 0;
    this->received_packet = nullptr;
    this->checksum_has_been_set = false;
}

void eddl_message::set_data(size_t message_data_size, void * data)
{
    this->message_data_size = message_data_size;

    if (0 == packet_data_size)
        throw std::runtime_error(err_msg("invalid packet_data_size."));

    this->seq_len = message_data_size / packet_data_size
                  + ((message_data_size % packet_data_size) != 0);

    if (0 == this->seq_len)
        throw std::runtime_error(err_msg("invalid seq_len."));

    this->pending_packets = this->seq_len;

    if (nullptr != this->data) free(this->data);
    this->data = (unsigned char *)eddl_malloc(message_data_size);
    if (nullptr == this->data)
        throw std::runtime_error(err_msg("error allocating memory."));
    if (nullptr != data) {
        memcpy(this->data, data, message_data_size);
        this->compute_checksum();
        this->checksum_has_been_set = true;
        this->pending_packets = 0;
    } else {
        memset(this->data, 0, message_data_size);
    }

    if (nullptr != this->received_packet) free(this->received_packet);
    this->received_packet = (bool *)eddl_malloc(this->seq_len * sizeof(bool));
    memset(this->received_packet, 0, this->seq_len * sizeof(bool));
}

eddl_message::~eddl_message()
{
    if (nullptr != this->data) free(this->data);
    if (nullptr != this->received_packet) free(this->received_packet);
}

uint32_t eddl_message::get_command()
{
    uint32_t *p = (uint32_t *)this->data;

    return *p;
}

void eddl_message::set_source_addr(uint32_t source_addr)
{
    this->source_addr = source_addr;
}
void eddl_message::set_target_addr(uint32_t target_addr)
{
    this->target_addr = target_addr;
}
void eddl_message::set_message_id(char * message_id)
{
    static char hex[20]="0123456789ABCDEF";

    if (nullptr != message_id) {
        if (strlen(message_id) < eddl_msg_id_len)
            throw std::runtime_error(err_msg("invalid message id"));

        this->message_id = std::string(message_id, eddl_msg_id_len);
    } else {
        char s[32];
        int i=0;

        uint32_t s_addr = this->source_addr;

        for (int k=0; k < 8; k++) {
            s[i++] = hex[s_addr & 0x00f];
            s_addr >>= 4;
        }

        uint32_t type = this->type;
        for (int k=0; k < 3; k++) {
            s[i++] = hex[type & 0x00f];
            type >>= 4;
        }

        uint64_t msec = this->timestamp;
        for (int k=0; k < 8; k++) {
            s[i++] = hex[msec & 0x00f];
            msec >>= 4;
        }
        s[i++] = '\0';

        this->message_id = s;
    }
}

void eddl_message::compute_checksum()
{
    SHA256((unsigned char *)this->data, this->message_data_size, this->checksum);
}
bool eddl_message::is_checksum_valid()
{
    unsigned char checksum[eddl_checksum_len];
    SHA256((unsigned char *)this->data, this->message_data_size, checksum);

    for (int i=0; i < eddl_checksum_len; i++)
        if (this->checksum[i] != checksum[i]) return false;

    return true;
}

void eddl_message::set_checksum(unsigned char * checksum)
{
    memcpy(this->checksum, checksum, eddl_checksum_len);
    this->checksum_has_been_set = true;
}

void eddl_message::add_packet(eddl_packet * packet)
{
    if (0 == this->packet_data_size)
        // this could fail if the first packet is the last one of the
        // sequence whose size is smaller than the remaining packet
        this->packet_data_size = packet->get_all_but_last_packet_size();

    if (nullptr == this->data) {
        set_data(packet->get_message_size(), nullptr);
    }

    size_t seq_no = packet->get_seq_no();

    if (seq_no >= this->seq_len)
        throw std::runtime_error(err_msg("invalid packet seq_no."));

    if (this->message_data_size != packet->get_message_size())
        throw std::runtime_error(err_msg("message_data_size discrepancy: "
                                + std::to_string(this->message_data_size)
                                + " vs "
                                + std::to_string(packet->get_message_size())));

    if (this->packet_data_size != packet->get_data_size()) {
        if (seq_no < this->seq_len-1)
            throw std::runtime_error(err_msg("packet_data_size discrepancy: "
                                    + std::to_string(this->packet_data_size)
                                    + " vs "
                                    + std::to_string(packet->get_data_size())));
        /*
        else
            print_err_msg("last packet of the message has a different data size.");
        */
    }

    size_t i = seq_no * this->packet_data_size;
    memcpy(&this->data[i], packet->get_data(), packet->get_data_size());

    if (! this->received_packet[seq_no]) {
        this->received_packet[seq_no] = true;
        this->pending_packets--;
    }
}
bool eddl_message::was_packet_already_added(size_t seq_no)
{
    if (seq_no >= this->seq_len)
        throw std::runtime_error(err_msg("invalid packet seq_no."));

    return this->received_packet != nullptr
        && this->received_packet[seq_no];
}

eddl_packet * eddl_message::get_packet(size_t packet_index)
{
    if (nullptr == this->data)
        throw std::runtime_error(err_msg("no data available."));

    if (packet_index >= this->seq_len)
        throw std::runtime_error(err_msg("invalid packet index."));

    size_t pos = this->packet_data_size * packet_index;

    if (pos >= this->message_data_size)
        throw std::runtime_error(err_msg("invalid index to access data."));

    size_t data_size_of_this_packet = std::min(this->packet_data_size,
                                               this->message_data_size - pos);

    return new eddl_packet(this->type,
                           this->source_addr,
                           this->target_addr,
                           this->message_id,
                           packet_index,
                           this->seq_len,
                           this->message_data_size,
                           this->packet_data_size,
                           data_size_of_this_packet,
                           & this->data[pos] );
}

eddl_packet * eddl_message::create_packet_for_checksum()
{
    return new eddl_packet(eddl_message_types::MSG_CHKSUM, //this->type,
                           this->source_addr,
                           this->target_addr,
                           this->message_id,
                           this->seq_len, // this is a special case where packet index is not used
                           this->seq_len,
                           this->message_data_size,
                           this->packet_data_size, // not sure if here it should be eddl_checksum_len
                           eddl_checksum_len,
                           this->checksum);
}

eddl_message * eddl_message::create_acknowledgement()
{
    uint32_t type = 0;
    switch (this->get_type()) {
        case eddl_message_types::DATA_SAMPLES:
            type = eddl_message_types::MSG_ACK_SAMPLES;
            break;
        case eddl_message_types::DATA_GRADIENTS:
            type = eddl_message_types::MSG_ACK_GRADIENTS;
            break;
        case eddl_message_types::DATA_WEIGHTS:
            type = eddl_message_types::MSG_ACK_WEIGHTS;
            break;
        default:
            throw std::runtime_error(err_msg("unexpected message type to be acknowledged."));
    }

    eddl_message * ack = new eddl_message(type,
                                          this->get_target_addr(), // source addr is the target addr of this message
                                          this->get_source_addr(), // target addr is the source addr of this message
                                          this->message_id.size(),
                                          this->message_id.size(),
                                          (void *)this->message_id.c_str() );
    return ack;
}
std::string eddl_message::get_acknowledged_message_id()
{
    return std::string((char *)this->data, eddl_msg_id_len);
}
uint32_t eddl_message::get_acknowledged_message_type()
{
    return ((this->data[ 8]-'0') << 8)
         + ((this->data[ 9]-'0') << 4)
         +  (this->data[10]-'0');
}


};
