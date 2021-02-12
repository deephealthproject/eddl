/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#ifndef __MulticastReceiver_H__
#define __MulticastReceiver_H__ 1

#include <vector>
#include <map>
#include <queue>
#include <thread>
#include <mutex>

#include <eddl/distributed/eddl_distributed.h>
#include <eddl/distributed/distributed_environment.h>
#include <eddl/distributed/eddl_queue.h>
#include <eddl/distributed/eddl_message.h>
#include <eddl/distributed/eddl_packet_ack.h>

namespace eddl {

class MulticastReceiver
{
public:
    MulticastReceiver(eddl_queue & input_queue,
                      eddl_queue & ack_queue,
                      eddl_queue & output_queue,
                      DistributedEnvironment & distributed_environment);
    ~MulticastReceiver();

    void stop();
    void receiver();
    void send_ack(eddl_packet_ack * ack);

private:
    eddl_queue &                input_queue;
    eddl_queue &                ack_queue;
    eddl_queue &                output_queue;
    DistributedEnvironment &    distributed_environment;

    int socket_fd_in; // input socket file descriptor
    int socket_fd_out; // output socket file descriptor
    int port_number_in; // input port number to use
    int port_number_out; // output port number to use

    uint32_t        my_s_addr;

    bool            receiver_active;
    std::thread     receiver_thread;

    std::map<std::string, eddl_message *>   active_messages;
    std::map<std::string, uint64_t>         recently_received_messages;
};

};

#endif // __MulticastReceiver_H__
