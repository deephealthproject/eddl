/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#ifndef __EDDL_MESSAGE_ACKS_H__
#define __EDDL_MESSAGE_ACKS_H__ 1

#include <string>
#include <map>

#include <eddl/distributed/eddl_distributed.h>
#include <eddl/distributed/eddl_message.h>
#include <eddl/distributed/eddl_worker_node.h>

namespace eddl {

class eddl_message_acks
{
public:
    eddl_message_acks(std::vector<eddl_worker_node *> & workers,
                      eddl_message * message);
    ~eddl_message_acks();

    void acknowledge(uint32_t source_addr, size_t seq_no);
    void acknowledge_whole_message(uint32_t source_addr);
    bool all_has_been_acknowledged();
    bool packet_already_acknowledged(size_t seq_no);
    bool lasting_too_much_time();
    ssize_t get_pending_acknowledgements();

private:
    std::map<uint32_t, int *>   acks;
    size_t                      living_workers;
    size_t                      num_acks_per_worker;
    size_t                      total_num_acks;
    size_t                      ack_counter;

    size_t *                    packet_counters;

    uint64_t                    starting_timestamp;
};

};

#endif // __EDDL_MESSAGE_ACKS_H__
