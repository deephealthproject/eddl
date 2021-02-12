/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#ifndef __EDDL_MESSAGE_H__
#define __EDDL_MESSAGE_H__ 1

#include <string>

#include <eddl/distributed/eddl_distributed.h>
#include <eddl/distributed/eddl_packet.h>

namespace eddl {

class eddl_message
{
public:
    eddl_message(uint32_t type,
                 uint32_t source_addr,
                 uint32_t target_addr,
                 size_t message_data_size,
                 size_t packet_data_size,
                 void * data );
    eddl_message(eddl_packet * packet);

    ~eddl_message();

    void set_data(size_t message_data_size, void * data);
    void set_source_addr(uint32_t source_addr);
    void set_target_addr(uint32_t target_addr);
    void set_message_id(char * message_id = nullptr);
    std::string & get_message_id() { return message_id; }
    std::string   get_acknowledged_message_id();
    uint32_t get_acknowledged_message_type();

    inline uint32_t get_type() { return type; }
    inline uint32_t get_source_addr() { return source_addr; }
    inline uint32_t get_target_addr() { return target_addr; }
    inline uint64_t get_timestamp() { return timestamp; }
    inline size_t   get_seq_len() { return seq_len; }
    inline bool     is_complete() { return 0 == pending_packets && checksum_has_been_set; }
    inline bool     was_checksum_already_set() { return checksum_has_been_set; }
    inline uint32_t get_message_data_size() { return message_data_size; }
    inline std::string get_checksum() { return std::string((char *)checksum, eddl_checksum_len); }
    inline unsigned char * get_checksum_ptr() { return checksum; }
    inline void * get_data() { return data; }
    std::string get_ip_address();

    uint32_t get_command();

    void compute_checksum();
    bool is_checksum_valid();

    void set_checksum(unsigned char * checksum);
    void add_packet(eddl_packet * packet);
    bool was_packet_already_added(size_t seq_no);
    eddl_packet * get_packet(size_t packet_index);
    eddl_packet * create_packet_for_checksum();

    eddl_message * create_acknowledgement();

    static eddl_message * start_command(uint32_t target_addr)
    {
        return new eddl_message(eddl_message_types::COMMAND,
                                0, target_addr,
                                0, eddl_command_types::START,
                                nullptr);
    }
    static eddl_message * stop_command(uint32_t target_addr)
    {
        return new eddl_message(eddl_message_types::COMMAND,
                                0, target_addr,
                                0, eddl_command_types::STOP,
                                nullptr);
    }
    static eddl_message * shutdown_command(uint32_t target_addr)
    {
        return new eddl_message(eddl_message_types::COMMAND,
                                0, target_addr,
                                0, eddl_command_types::SHUTDOWN,
                                nullptr);
    }
    static eddl_message * acknowledgement(eddl_packet * packet, uint32_t my_s_addr)
    {
        size_t data[2];
        data[0] = packet->get_seq_no();
        data[1] = packet->get_type();

        return new eddl_message(eddl_message_types::PKG_ACK,
                                my_s_addr,
                                packet->get_source_addr(), // acknowledgement must be sent to the packet emisor
                                sizeof(data),
                                sizeof(data),
                                &data);
    }

private:
    uint32_t        type;
    uint32_t        source_addr;
    uint32_t        target_addr;
    uint64_t        timestamp;
    size_t          seq_len;
    size_t          message_data_size;
    size_t          packet_data_size;
    std::string     message_id;
    unsigned char   checksum[eddl_checksum_len];
    unsigned char * data;

    size_t          pending_packets;
    bool *          received_packet;
    bool            checksum_has_been_set;
};

};

#endif // __EDDL_MESSAGE_H__
