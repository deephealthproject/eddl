/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: August 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#ifndef __MULTICAST_SENDER_H__
#define __MULTICAST_SENDER_H__ 1

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <vector>
#include <map>
#include <queue>
#include <thread>
#include <mutex>

#include <eddl/distributed/eddl_distributed.h>
#include <eddl/distributed/distributed_environment.h>
#include <eddl/distributed/eddl_queue.h>
#include <eddl/distributed/eddl_message.h>
#include <eddl/distributed/eddl_message_acks.h>
#include <eddl/distributed/eddl_packet_ack.h>
#include <eddl/distributed/eddl_worker_node.h>

namespace eddl {

class MulticastSender
{
public:
    MulticastSender(std::vector<eddl_worker_node *> & workers,
                    eddl_queue & output_queue,
                    eddl_queue & weights_ack_queue,
                    DistributedEnvironment & distributed_environment);
    ~MulticastSender();

    void stop();
    void sender();
    void ack_processor();
    bool send_message(eddl_message * message);

private:
    std::vector<eddl_worker_node *> &   workers;
    eddl_queue &                        output_queue;
    eddl_queue &                        weights_ack_queue;
    DistributedEnvironment &            distributed_environment;
/*
    std::string     multicast_group_addr;
    int             port_number_out; // input port number to use
    int             port_number_in; // input port number to use
*/
    int             socket_fd_in; // input socket file descriptor
    int             socket_fd_out; // output socket file descriptor

    struct sockaddr_in  target_group_addr;

    bool            sender_active;
    std::thread     sender_thread;
    std::thread     ack_processor_thread;
    std::mutex      ack_processor_mutex;

    size_t          sent_bytes_threshold;

    std::map<std::string, eddl_message_acks *>  active_acknowledgements;
}; // class MulticastSender

}; // namespace eddl

#endif // __MULTICAST_SENDER_H__
