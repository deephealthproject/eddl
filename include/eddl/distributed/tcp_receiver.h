/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#ifndef __TCP_RECEIVER_H__
#define __TCP_RECEIVER_H__ 1

#include <vector>
#include <queue>
#include <thread>
#include <mutex>

#include <eddl/distributed/eddl_distributed.h>
#include <eddl/distributed/distributed_environment.h>
#include <eddl/distributed/eddl_queue.h>

namespace eddl {

class TCP_Receiver
{
private:
    class ActiveThread
    {
        public:
            ActiveThread(int socket_fd,
                         eddl_queue & input_queue,
                         eddl_queue & weights_ack_queue,
                         eddl_queue & generic_ack_queue,
                         eddl_queue & output_queue,
                         TCP_Receiver * tcp_receiver);
            ~ActiveThread();
            inline eddl_thread_status get_status() { return this->status; }
            inline void stop() { this->status=STOPPED; }
            inline void disable() { this->status=INACTIVE; }
            void join() { this->thread->join(); }
            eddl_message * receive_message();
            void thread_receiver();

        private:
            std::thread *       thread;
            eddl_thread_status  status;
            int                 socket_fd;
            eddl_queue  &       input_queue;
            eddl_queue  &       weights_ack_queue;
            eddl_queue  &       generic_ack_queue;
            eddl_queue  &       output_queue;
            TCP_Receiver *      tcp_receiver;
    };

public:
    TCP_Receiver(   eddl_queue & input_queue,
                    eddl_queue & weights_ack_queue,
                    eddl_queue & generic_ack_queue,
                    eddl_queue & output_queue,
                    DistributedEnvironment & distributed_environment);
    ~TCP_Receiver();

    void stop();
    void drop_stopped();
    void joiner();
    void acceptor();

private:
    eddl_queue &                input_queue;
    eddl_queue &                weights_ack_queue;
    eddl_queue &                generic_ack_queue;
    eddl_queue &                output_queue;
    DistributedEnvironment &    distributed_environment;

    int             socket_fd; // socket file descriptor
    bool            receiver_active;
    std::thread     joiner_thread;
    std::thread     acceptor_thread;

    std::queue<ActiveThread *>  active_threads;
    std::mutex                  mutex_active_threads;
};

};

#endif // __TCP_RECEIVER_H__
