/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <string>
#include <iostream>

#include <eddl/distributed/tcp_sender.h>

namespace eddl {


TCP_Sender::TCP_Sender(eddl_queue & output_queue,
                       eddl_queue & ack_queue,
                       DistributedEnvironment & distributed_environment) :
    output_queue(output_queue),
    ack_queue(ack_queue),
    distributed_environment(distributed_environment)
{
    change_status_to(NORMAL_OPERATION);
    sender_active = true;
    sender_thread = std::thread( & TCP_Sender::sender, this);
}

void TCP_Sender::change_status_to(int new_status)
{
    this->sender_status = new_status;
    this->timestamp_last_status_change = get_system_milliseconds();
}

TCP_Sender::~TCP_Sender()
{
    stop();
    sender_thread.join();

    /*
        see the code of eddl_queue, it wipes itself by deleting the pending messages,
        so it is not necessary to delete the messages in any of the three queues:
            output_queue
            ack_queue
            queue_of_pending_messages
    */

    for (auto &x : sent_messages) {
        print_err_msg("deleting message with id "
                    + x.first
                    + " pending to be acknowledged.");
        delete x.second;
    }
    sent_messages.clear();
}
void TCP_Sender::stop()
{
    sender_active = false;
    queue_of_pending_messages.clear();
    // the output queue must be cleared in the main thread of a worker or the master
}

void TCP_Sender::sender()
{
    while (sender_active) {
        eddl_message * message = nullptr;

        if (sender_status == NORMAL_OPERATION) {
            // messages in the queue of pending messages have more priority
            while (sender_active  &&  nullptr == message  &&  ! queue_of_pending_messages.empty()) {
                // the pop() method blocks and waits until data is ready
                message = queue_of_pending_messages.pop();
                if (get_system_milliseconds() - message->get_timestamp() > 50000 /* 50 seconds */) {
                    // too old messages are dropped
                    print_err_msg("dropping too old message " + message->get_message_id()
                                + " from the queue of pending messages!");
                    delete message;
                    message = nullptr;
                }
            }
            // only gets messages from the output queue if the queue of pending messages is empty
            if (nullptr == message  &&  ! output_queue.empty()) {
                // the pop() method blocks and waits until data is ready
                message = output_queue.pop();
            }

            if (nullptr != message) {
                manage_to_send_message(message);
            } else {
                if (ack_queue.empty())
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } else {
            if (get_system_milliseconds()-this->timestamp_last_status_change > 1000) {
                change_status_to(NORMAL_OPERATION);
            }
        }
        while (! ack_queue.empty()) {
            eddl_message * ack = ack_queue.pop();
            std::string msg_id = ack->get_acknowledged_message_id();
            if (sent_messages.count(msg_id) > 0) {
                delete sent_messages.at(msg_id);
                sent_messages.erase(msg_id);
                /*
                if (verbose_level >= 2)
                    print_log_msg("sent_messages[" + msg_id + "]  ERASED"
                          + "   |sent_messages| = " + std::to_string(sent_messages.size()));
                */
            }
            delete ack;
        }
    }
    print_log_msg("sender thread stopped normally.");
}

void TCP_Sender::manage_to_send_message(eddl_message * message)
{
    if (send_message(message)) {
        /*
            a sent message is maintained in sent_messages map in order to wait
            for the corresponding acknowledgement, when the acknowledgement
            of a message is received, then the message is removed
        */
        std::string msg_id;
        switch (message->get_type()) {
            case eddl_message_types::DATA_SAMPLES:
            case eddl_message_types::DATA_GRADIENTS:
            case eddl_message_types::DATA_WEIGHTS:
                msg_id = message->get_message_id();
                /*
                if (verbose_level >= 2)
                    print_err_msg("sent_messages[" + msg_id + "]  INSERTED"
                          + "   |sent_messages| = " + std::to_string(sent_messages.size()+1));
                */
                if (sent_messages.count(msg_id) > 0) {
                    throw std::runtime_error(err_msg("recently sent message already existed in the map."));
                }
                sent_messages[msg_id] = message;
                break;

            default :
                delete message;
                break;
        }
    } else if (sender_active) {
        queue_of_pending_messages.push(message);
    } else {
        delete message;
    }
}
bool TCP_Sender::send_message(eddl_message * message)
{
    int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0)
        throw std::runtime_error(err_msg("socket cannot be created."));

    struct sockaddr_in  peer_addr;

    memset(&peer_addr, 0, sizeof(struct sockaddr_in));
    peer_addr.sin_family = AF_INET;
    peer_addr.sin_addr.s_addr = message->get_target_addr();
    peer_addr.sin_port = htons(distributed_environment.get_tcp_port());

    /*
    if (verbose_level >= 1)
        print_log_msg("trying to connect to " + inet_ntoa(peer_addr.sin_addr));
    */

    if (connect(socket_fd, (const sockaddr *)&peer_addr, sizeof(peer_addr)) < 0) {
        close(socket_fd);
        print_err_msg("failed to connect.");
        change_status_to(FAILED_TO_CONNECT);
        return false;
    }

    message->set_source_addr(distributed_environment.get_my_s_addr());
    // compulsory to compute again the message id every time source addr is updated
    message->set_message_id();

    uint32_t type = message->get_type();
    char msg_id[eddl_msg_id_len+1]; strncpy(msg_id, message->get_message_id().c_str(), eddl_msg_id_len);
    uint32_t source_addr = distributed_environment.get_my_s_addr();
    uint32_t target_addr = message->get_target_addr();
    unsigned char * checksum = message->get_checksum_ptr();
    size_t size_in_bytes = message->get_message_data_size();

    ssize_t n, s;
    size_t l;

    try {
        // send the message type
        s = l = sizeof(type);
        n = write(socket_fd, &type, l);
        if (n != s) {
            close(socket_fd);
            print_err_msg("failed to send message type.");
            change_status_to(FAILED_TO_WRITE);
            return false;
        }

        // send the message id
        s = l = eddl_msg_id_len;
        n = write(socket_fd, &msg_id, l);
        if (n != s) {
            close(socket_fd);
            print_err_msg("failed to send message id.");
            change_status_to(FAILED_TO_WRITE);
            return false;
        }

        // send the message sender s_addr
        s = l = sizeof(source_addr);
        n = write(socket_fd, &source_addr, l);
        if (n != s) {
            close(socket_fd);
            print_err_msg("failed to send sender s_addr.");
            change_status_to(FAILED_TO_WRITE);
            return false;
        }

        // send the message receiver s_addr
        s = l = sizeof(target_addr);
        n = write(socket_fd, &target_addr, l);
        if (n != s) {
            close(socket_fd);
            print_err_msg("failed to send receiver s_addr.");
            change_status_to(FAILED_TO_WRITE);
            return false;
        }

        // send the message checksum
        s = l = eddl_checksum_len;
        n = write(socket_fd, checksum, l);
        if (n != s) {
            close(socket_fd);
            print_err_msg("failed to message checksum.");
            change_status_to(FAILED_TO_WRITE);
            return false;
        }

        // send the message size in bytes
        s = l = sizeof(size_in_bytes);
        n = write(socket_fd, &size_in_bytes, l);
        if (n != s) {
            close(socket_fd);
            print_err_msg("failed to message size.");
            change_status_to(FAILED_TO_WRITE);
            return false;
        }

        // send the message data
        size_t block_size = eddl_default_mtu;
        size_t pending = size_in_bytes;
        char * ptr = (char *)message->get_data();

        while( pending > 0 ) {

            s = l = (pending < block_size) ? pending : block_size;

            n = write(socket_fd, ptr, l);
            if (n < 0) {
                std::string str = "n = " + std::to_string(n)
                              + " errno = " + std::to_string(errno)
                              + " " + strerror(errno)
                              + " ptr = " + pointer_to_string(ptr)
                              + " bytes received = " + std::to_string(ptr - (char *)message->get_data());
                print_err_msg("write failed  " + str);
                close(socket_fd);
                change_status_to(FAILED_TO_WRITE);
                return false;
            }

            if (n < s) std::this_thread::sleep_for(std::chrono::milliseconds(10));

            pending -= n;
            ptr += n;
        }
    }
    catch(std::exception & e) {
        print_err_msg("exception ocurred: " + std::string(e.what()));
        close(socket_fd);
        change_status_to(FAILED_TO_WRITE);
        return false;
    }
    close(socket_fd);
    return true;
}

};
