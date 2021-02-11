/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#include <cstring>
#include <csignal>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <iostream>
#include <exception>

#include <eddl/distributed/tcp_receiver.h>

namespace eddl {


TCP_Receiver::TCP_Receiver( eddl_queue & input_queue,
                            eddl_queue & weights_ack_queue,
                            eddl_queue & generic_ack_queue,
                            eddl_queue & output_queue,
                            DistributedEnvironment & distributed_environment) :
    input_queue(input_queue),
    weights_ack_queue(weights_ack_queue),
    generic_ack_queue(generic_ack_queue),
    output_queue(output_queue),
    distributed_environment(distributed_environment)
{
    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0)
        throw std::runtime_error(err_msg("socket cannot be created."));

    struct sockaddr_in  my_addr;

    /* Clear data structure */
    memset(&my_addr, 0, sizeof(struct sockaddr_in));
    my_addr.sin_family = AF_INET;
    my_addr.sin_addr.s_addr = INADDR_ANY;
    my_addr.sin_port = htons(distributed_environment.get_tcp_port());

    if (bind(socket_fd, (struct sockaddr *) &my_addr, sizeof(struct sockaddr_in)) < 0)
        throw std::runtime_error(err_msg("binding socket failed."));

    if (listen(socket_fd, listen_max_pending) < 0)
        throw std::runtime_error(err_msg("setting listening state failed."));

    signal(SIGPIPE, SIG_IGN);

    receiver_active=true;
    joiner_thread   = std::thread( & TCP_Receiver::joiner,   this);
    acceptor_thread = std::thread( & TCP_Receiver::acceptor, this);
}

TCP_Receiver::~TCP_Receiver()
{
    stop();
    receiver_active=false;

    // a signal must be sent to the acceptor thread in order to wake up it
    // from the accept() system call, but the solution is to deatch the thread
    // and leave the program to end, then the thread is killed.

    close(socket_fd);

    joiner_thread.join();
    //acceptor_thread.detach();
    acceptor_thread.join(); // hoping the master sends the stopping commands

    drop_stopped();

    /*
        an object of this class can be destroyed and this destructor complete
        its execution while one (or more) objects of the class ActiveThread
        remain(s) active while reading an incoming message.
        if this occurs, then the master process monitored by valgrind will reports
        some reachable memory blocks.
    */
}

void TCP_Receiver::stop()
{
    receiver_active=false;
}
void TCP_Receiver::acceptor()
{
    while (receiver_active) {

        struct sockaddr_in  peer_addr;
        socklen_t peer_addr_size = sizeof(struct sockaddr_in);

        int connected_socket_fd = accept(socket_fd, (struct sockaddr *) &peer_addr, &peer_addr_size);

        if (connected_socket_fd < 0) {
            if (receiver_active)
                throw std::runtime_error(err_msg("accepting a connection failed."));
        } else {
            /*
            if (verbose_level >= 1)
                print_log_msg("connection accepted from " + inet_ntoa(peer_addr.sin_addr));
            */

            ActiveThread * at = new ActiveThread(connected_socket_fd, input_queue, weights_ack_queue, generic_ack_queue, output_queue, this);

            { // critical region for pushing new items in the queue of active threads
                std::unique_lock<std::mutex> lck(mutex_active_threads);

                active_threads.push(at);
            }
        }
    }
    print_log_msg("acceptor thread stopped normally.");
}

void TCP_Receiver::drop_stopped()
{   /*
        this method is wholly executed inside a critical region that
        takes exclusive access to the queue of active_threads
    */
    // Critical region starts
    std::unique_lock<std::mutex> lck(mutex_active_threads);

    for (unsigned int i=0; i < active_threads.size(); i++) {

        // pops an active thread from queue
        ActiveThread * at = active_threads.front(); active_threads.pop();

        if (at->get_status() == STOPPED) {
            // if the active thread is stopped then it is joined and destroyed
            at->join();
            at->disable();
            delete at;
        } else {
            // otherwise it is pushed again into the queue
            active_threads.push(at);
        }
    }
    // Critical region ends
}
void TCP_Receiver::joiner()
{
    while (receiver_active) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        drop_stopped();
    }

    print_log_msg("joiner thread stopped normally.");
}



/////////////////////////////////////////////////////////////////////////////
///////////////////// METHODS OF CLASS ActiveThread ////////////////////////
/////////////////////////////////////////////////////////////////////////////
TCP_Receiver::ActiveThread::ActiveThread(int socket_fd,
                                         eddl_queue & input_queue,
                                         eddl_queue & weights_ack_queue,
                                         eddl_queue & generic_ack_queue,
                                         eddl_queue & output_queue,
                                         TCP_Receiver * tcp_receiver)
:   socket_fd(socket_fd),
    input_queue(input_queue),
    weights_ack_queue(weights_ack_queue),
    generic_ack_queue(generic_ack_queue),
    output_queue(output_queue),
    tcp_receiver(tcp_receiver)
{
    status = INACTIVE;
    thread = new std::thread( & ActiveThread::thread_receiver, this);
}
TCP_Receiver::ActiveThread::~ActiveThread()
{
    /*
        here it is not necessary to join the thread because
        it is assumed here the thread was joined from the
        drop_stopped() method of the class TCP_Receiver
    */
    delete thread;
}
void TCP_Receiver::ActiveThread::thread_receiver()
{
    this->status = RUNNING;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    eddl_message * message = receive_message();
    close(socket_fd);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    int msec = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    if (nullptr != message) {
        if (tcp_receiver->distributed_environment.get_verbose_level() >= 1)
            print_log_msg("thread on socket " + std::to_string(socket_fd)
                    + " completed after receiving " + std::to_string(message->get_message_data_size())
                    + " bytes in " + std::to_string(msec/1.0e6) + " seconds!"
                    + "  message_type: " + get_message_type_name(message->get_type()));

        switch (message->get_type()) {

            case eddl_message_types::DATA_SAMPLES:
            case eddl_message_types::DATA_GRADIENTS:
            case eddl_message_types::DATA_WEIGHTS:
                output_queue.push(message->create_acknowledgement());
                input_queue.push(message);
                break;

            case eddl_message_types::COMMAND:
                if (message->get_command() == eddl_command_types::SHUTDOWN)
                    this->tcp_receiver->receiver_active = false;
            case eddl_message_types::PARAMETER:
                input_queue.push(message);
                break;

            case eddl_message_types::MSG_ACK_WEIGHTS:
                weights_ack_queue.push(message);
                break;

            case eddl_message_types::MSG_ACK_GRADIENTS:
            case eddl_message_types::MSG_ACK_SAMPLES:
                generic_ack_queue.push(message);
                break;

            case eddl_message_types::PKG_ACK:
                {
                    size_t * p = (size_t *)message->get_data();
                    // in p[1] is the type of the acknowledged message
                    // see method acknowledgement(eddl_packet *)
                    // in file eddl_message.h
                    switch (p[1]) {
                        case eddl_message_types::DATA_WEIGHTS:
                            weights_ack_queue.push(message);
                            break;
                        default:
                            generic_ack_queue.push(message);
                            break;
                    }
                }
                break;

            default:
                throw std::runtime_error(err_msg("non-expected message type"));
        }
    } else {
        print_err_msg("thread on socket " + std::to_string(socket_fd)
                + " received an erroneous message in "
                + std::to_string(msec/1.0e6) + " seconds!");
    }
    this->status = STOPPED;
}
eddl_message * TCP_Receiver::ActiveThread::receive_message()
{
    uint32_t type;
    char     msg_id[eddl_msg_id_len+1];
    size_t   size_in_bytes;
    size_t   block_size = eddl_default_mtu;
    uint32_t source_addr, target_addr;
    unsigned char checksum[eddl_checksum_len];

    ssize_t n, s;
    size_t l;

    eddl_message * message = nullptr;

    try {
        // receive message type
        s = l = sizeof(type);
        n = read(socket_fd, &type, l);
        if (n != s) { print_err_msg("message type read failed."); return nullptr; }

        // receive message id
        memset(msg_id, 0, eddl_msg_id_len+1);
        s = l = eddl_msg_id_len;
        n = read(socket_fd, msg_id, l);
        if (n != s) { print_err_msg("message id read failed."); return nullptr; }

        // receive message sender s_addr
        s = l = sizeof(source_addr);
        n = read(socket_fd, &source_addr, l);
        if (n != s) { print_err_msg("message sender s_addr read failed."); return nullptr; }

        // receive message target s_addr
        s = l = sizeof(target_addr);
        n = read(socket_fd, &target_addr, l);
        if (n != s) { print_err_msg("message receiver s_addr read failed."); return nullptr; }

        // receive message checksum
        memset(checksum, 0, eddl_checksum_len); // otherwise valgrind reports a warning
        s = l = eddl_checksum_len;
        n = read(socket_fd, checksum, l);
        if (n != s) { print_err_msg("message checksum read failed."); return nullptr; }

        // receive message size_in_bytes
        s = l = sizeof(size_in_bytes);
        n = read(socket_fd, &size_in_bytes, l);
        if (n != s) { print_err_msg("message size in bytes read failed."); return nullptr; }

        message = new eddl_message(type,
                                   source_addr,
                                   target_addr,
                                   size_in_bytes,
                                   eddl_packet_data_size,
                                   nullptr);

        message->set_message_id(msg_id);
        message->set_checksum(checksum);

        char * ptr = (char *)message->get_data();
        size_t bytes_received=0;

        /*
        if (verbose_level >= 1)
            print_log_msg("receiving a data message of " + std::to_string(size_in_bytes) + " bytes");
        */

        while( bytes_received < size_in_bytes ) {

            l = size_in_bytes - bytes_received;
            l = (l < block_size) ? l : block_size;

            n = read(socket_fd, ptr, l);
            if (n < 0)
                throw std::runtime_error(err_msg(std::string("read data block failed:")
                                       + std::to_string(errno) + ":" + strerror(errno)));

            if (n == 0) break;

            bytes_received += n;
            ptr += n;
        }
    }
    catch(std::exception & e) {
        print_err_msg("an exception ocurred: " + std::string(e.what()));
        delete message;
        return nullptr;
    }

    if (message->is_checksum_valid()) {
        return message;
    } else {
        delete message;
        return nullptr;
    }
}

};
