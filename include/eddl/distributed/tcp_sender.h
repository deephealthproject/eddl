/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#ifndef __TCP_SENDER_H__
#define __TCP_SENDER_H__ 1

#include <vector>
#include <queue>
#include <map>
#include <thread>
#include <mutex>

#include <eddl/distributed/distributed_environment.h>
#include <eddl/distributed/eddl_distributed.h>
#include <eddl/distributed/eddl_message.h>
#include <eddl/distributed/eddl_queue.h>

namespace eddl {

class TCP_Sender
{
public:
    TCP_Sender(eddl_queue & output_queue,
               eddl_queue & ack_queue,
               DistributedEnvironment & distributed_environment);
    ~TCP_Sender();

    void stop();
    void sender();
    bool send_message(eddl_message * msg);
    void manage_to_send_message(eddl_message * msg);

    void change_status_to(int new_status);

private:
    eddl_queue &                            output_queue;
    eddl_queue &                            ack_queue;
    DistributedEnvironment &                distributed_environment;
    bool                                    sender_active;
    std::thread                             sender_thread;
    eddl_queue                              queue_of_pending_messages;
    std::map<std::string, eddl_message *>   sent_messages;
    int                                     sender_status;
    uint64_t                                timestamp_last_status_change;

    static constexpr int                    NORMAL_OPERATION=0;
    static constexpr int                    FAILED_TO_CONNECT=1;
    static constexpr int                    FAILED_TO_WRITE=2;
};

};

#endif // __TCP_SENDER_H__
