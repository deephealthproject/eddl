/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: August 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#include <eddl/distributed/eddl_message_acks.h>

#include <iostream>

namespace eddl {

eddl_message_acks::eddl_message_acks(std::vector<eddl_worker_node *> & workers,
                                     eddl_message * message)
{
    // we need space for the checksum show seq_no is equal to seq_len
    this->num_acks_per_worker = message->get_seq_len()+1;
    this->living_workers = 0;
    for (auto w: workers) {
        if (w->is_active()) {
            int * ptr = new int [this->num_acks_per_worker];
            memset(ptr, 0, this->num_acks_per_worker * sizeof(int));
            this->acks[w->get_s_addr()] = ptr;
            this->living_workers++;
        }
    }
    this->total_num_acks = this->num_acks_per_worker * this->living_workers;
    this->ack_counter = 0;

    this->packet_counters = new size_t [this->num_acks_per_worker];
    memset(this->packet_counters, 0, this->num_acks_per_worker * sizeof(size_t));

    this->starting_timestamp = get_system_milliseconds();
}
eddl_message_acks::~eddl_message_acks()
{
    for (auto iter : this->acks)
        delete [] iter.second;

    delete [] this->packet_counters;
}

ssize_t eddl_message_acks::get_pending_acknowledgements()
{
    return this->total_num_acks - this->ack_counter;
}

void eddl_message_acks::acknowledge(uint32_t source_addr, size_t seq_no)
{
    if (seq_no >= this->num_acks_per_worker)
        throw std::runtime_error(err_msg("invalid seq_no"));

    if (this->acks.count(source_addr) == 0)
        throw std::runtime_error(err_msg("invalid source_addr " + get_ip_address(source_addr)));

    if (this->acks[source_addr][seq_no] == 0) {
        this->acks[source_addr][seq_no] = 1;
        this->ack_counter++;
        this->packet_counters[seq_no]++;
    }
}
void eddl_message_acks::acknowledge_whole_message(uint32_t source_addr)
{
    if (this->acks.count(source_addr) == 0)
        throw std::runtime_error(err_msg("invalid source_addr " + get_ip_address(source_addr)));

    for(size_t seq_no = 0; seq_no < this->num_acks_per_worker; seq_no++) {
        if (this->acks[source_addr][seq_no] == 0) {
            this->acks[source_addr][seq_no] = 1;
            this->ack_counter++;
            this->packet_counters[seq_no]++;
        }
    }
}
bool eddl_message_acks::all_has_been_acknowledged()
{
    return this->ack_counter == this->total_num_acks;
}
bool eddl_message_acks::packet_already_acknowledged(size_t seq_no)
{
    if (seq_no >= this->num_acks_per_worker)
        throw std::runtime_error(err_msg("invalid seq_no"));

    return this->packet_counters[seq_no] == this->living_workers;
}

bool eddl_message_acks::lasting_too_much_time()
{
    // returns true if more than 60 seconds to be acknowledged
    return (get_system_milliseconds() - this->starting_timestamp) > 60*1000;
}

};
