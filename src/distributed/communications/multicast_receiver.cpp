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

#include <iostream>

#include <eddl/distributed/multicast_receiver.h>
#include <eddl/distributed/eddl_packet.h>

#if !defined(MSG_NOSIGNAL)
#   if defined(__APPLE__)
#       define MSG_NOSIGNAL 0
#   else
#       error "MSG_NOSIGNAL is not defined this should be fixed!"
#   endif
#endif

namespace eddl {


MulticastReceiver::MulticastReceiver(eddl_queue & input_queue,
                                     eddl_queue & ack_queue,
                                     eddl_queue & output_queue,
                                     DistributedEnvironment & distributed_environment) :
    input_queue(input_queue),
    ack_queue(ack_queue),
    output_queue(output_queue),
    distributed_environment(distributed_environment)
{
    socket_fd_in = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_in < 0)
        throw std::runtime_error(err_msg("input socket cannot be created."));

    int reuse=1;
    if (setsockopt(socket_fd_in, SOL_SOCKET, SO_REUSEADDR, (char *)&reuse, sizeof(reuse)) < 0)
        throw std::runtime_error(err_msg("socket cannot be set to allow multiple instances to receive copies of multicast datagrams."));

#if defined(__APPLE__)
    {
        int set = 1;
        if (setsockopt(socket_fd_in, SOL_SOCKET, SO_NOSIGPIPE, (void *)&set, sizeof(int)) < 0)
            throw std::runtime_error(err_msg("cannot unset SIGPIPE. " + std::to_string(errno) + ":" + strerror(errno)));
    }
#endif

    this->port_number_in = distributed_environment.get_udp_data_port();

    struct sockaddr_in  host_addr;
    memset(&host_addr, 0, sizeof(host_addr));
    host_addr.sin_family = AF_INET;
    host_addr.sin_addr.s_addr = INADDR_ANY; // distributed_environment.get_my_s_addr();
    host_addr.sin_port = htons(this->port_number_in);

    if (bind(socket_fd_in, (struct sockaddr *) &host_addr, sizeof(host_addr)) < 0)
        throw std::runtime_error(err_msg("binding socket failed."));

    struct ip_mreq  mreq;
    mreq.imr_multiaddr.s_addr = distributed_environment.get_multicast_s_addr();
    mreq.imr_interface.s_addr = distributed_environment.get_my_s_addr();

    if (setsockopt(socket_fd_in, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *) &mreq, sizeof(mreq)) < 0)
        throw std::runtime_error(err_msg("adding membership to multicast group failed."));

    ////////////////////////////////////////////////////////////////////////////

    socket_fd_out = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_out < 0)
        throw std::runtime_error(err_msg("output socket cannot be created."));

#if defined(__APPLE__)
    {
        int set = 1;
        if (setsockopt(socket_fd_out, SOL_SOCKET, SO_NOSIGPIPE, (void *)&set, sizeof(int)) < 0)
            throw std::runtime_error(err_msg("cannot unset SIGPIPE. " + std::to_string(errno) + ":" + strerror(errno)));
    }
#endif

    this->port_number_out = distributed_environment.get_udp_ack_port();

    ////////////////////////////////////////////////////////////////////////////

    std::cout   << "ready to receive messages from multicast group "
                << get_ip_address(mreq.imr_multiaddr.s_addr) << ":" << this->port_number_in
                << " via " << distributed_environment.get_my_ip_addr()
                << " and sent acknowledgements to "
                << distributed_environment.get_master_ip_addr() << ":" << this->port_number_out
                << std::endl;

    ////////////////////////////////////////////////////////////////////////////

    receiver_active = true;
    receiver_thread = std::thread( & MulticastReceiver::receiver, this);
}

MulticastReceiver::~MulticastReceiver()
{
    receiver_active = false;
    receiver_thread.join();
    recently_received_messages.clear();
    for (auto iter : active_messages)
        delete iter.second;
    active_messages.clear();
    close(socket_fd_out);
}

void MulticastReceiver::stop()
{
    receiver_active = false;
    // does this method to send a packet with closing command in order to
    // unlock the receiver thread?
}

void MulticastReceiver::send_ack(eddl_packet_ack * ack)
{
    int flags = MSG_NOSIGNAL;
    struct sockaddr_in  peer_addr;
    memset(&peer_addr, 0, sizeof(peer_addr));
    peer_addr.sin_family = AF_INET;
    peer_addr.sin_addr.s_addr = distributed_environment.get_master_s_addr();
    peer_addr.sin_port = htons(this->port_number_out);

    ssize_t l = sizeof(eddl_packet_ack);
    ssize_t n = sendto(socket_fd_out, (void *)ack, l, flags,
                        (const struct sockaddr *)&peer_addr, sizeof(peer_addr));
    //print_log_msg("sent acknowledgement of packet no " + std::to_string(ack->get_seq_no()));
    delete ack;

    if (n != l)
        throw std::runtime_error(err_msg("sent " + std::to_string(n)
                                       + " bytes instead of " + std::to_string(l)
                                       + "  " + std::to_string(errno) + ": "
                                       + strerror(errno)));
}
void MulticastReceiver::receiver()
{
    void * data;
    while (receiver_active) {
        struct sockaddr_in  peer_addr;
        socklen_t peer_addr_size = sizeof(peer_addr);
        int flags = MSG_NOSIGNAL; // MSG_WAITALL;
        data = eddl_malloc(sizeof(eddl_packet));
        if (nullptr == data)
            throw std::runtime_error(err_msg("error allocating memory."));

        ssize_t l = sizeof(eddl_packet);
        // blocking call
        ssize_t n = recvfrom(socket_fd_in, data, l, flags,
                             (struct sockaddr *)&peer_addr, &peer_addr_size);
        if (n < 0) {
            print_err_msg("error receiving a packet: " + std::to_string(errno) + ": " + strerror(errno));
            free(data);
            continue; // do not abort the process, just drop the packet
        }

        if (n != l) {
            print_err_msg("warning received an incomplete packet of "
                        + std::to_string(n) + " bytes instead of "
                        + std::to_string(l) + " bytes requrested");
            free(data);
            continue; // do not abort the process, just drop the packet
        }

        eddl_packet * packet = (eddl_packet *)data;
        /**
        print_log_msg("received packet " + std::to_string(packet->get_seq_no())
                    + "/" + std::to_string(packet->get_seq_len())
                    + " of message " + packet->get_message_id()
                    + " from " + get_ip_address(packet->get_source_addr()));
        **/
        if (packet->is_checksum_valid()) {
            if (packet->get_source_addr() != peer_addr.sin_addr.s_addr)
                throw std::runtime_error(err_msg("received packet from "
                                        + get_ip_address(peer_addr.sin_addr.s_addr)
                                        + " claiming it was sent from "
                                        + get_ip_address(packet->get_source_addr())));

            eddl_message * message = nullptr;
            std::string msg_id = "";

            switch(packet->get_type()) {
                case eddl_message_types::DATA_WEIGHTS:
                case eddl_message_types::DATA_GRADIENTS:
                case eddl_message_types::DATA_SAMPLES:
                case eddl_message_types::MSG_CHKSUM:
                    /* get info from packet and add it in the corresponding
                       existing message or just create a new message,
                       but if the messsage was recently received then resent
                       packets must be ignored (dropped)
                    */

                    /* contrary to the above comment, the acknowledgement is sent
                       in any case in order to allow multicast_sender in the peer
                       to know the packet was received -- this is pending to be
                       analysed in more detail, but currently this seems that
                       alevaites the problem of pending messages to be sent in the
                       multicast sender of the peer.
                    */
                    this->send_ack(packet->create_acknowledgement(distributed_environment.get_my_s_addr()));

                    msg_id = packet->get_message_id();
                    if (this->recently_received_messages.count(msg_id) > 0) {
                        uint64_t lapse = get_system_milliseconds() - this->recently_received_messages[msg_id];
                        // if more than one hour it was received then remove it
                        if (lapse > 1*60*60*1000) {
                            this->recently_received_messages.erase(msg_id);
                        }
                    } else {
                        if (this->active_messages.count(packet->get_message_id()) == 0) {
                            message = new eddl_message(packet->get_type(),
                                                       packet->get_source_addr(),
                                                       packet->get_target_addr(),
                                                       packet->get_message_size(),
                                                       packet->get_data_size(),
                                                       (void *)nullptr);
                            message->set_message_id(packet->get_message_id_ptr());
                            this->active_messages[message->get_message_id()] = message;
                            //print_log_msg(std::string("receiving message ") + message->get_message_id());
                            //print_log_msg(".....................................................message created from packet with " + std::to_string(packet->get_message_size()) + " vs " + std::to_string(message->get_message_data_size()));
                        } else {
                            message = this->active_messages[packet->get_message_id()];
                        }
                        if (packet->get_type() == eddl_message_types::MSG_CHKSUM) {
                            //print_log_msg(".....................................................message checksum received");
                            if (!message->was_checksum_already_set()) {
                                message->set_checksum((unsigned char *)packet->get_data());
                            }
                            //  this->send_ack(packet->create_acknowledgement(distributed_environment.get_my_s_addr()));
                        } else {
                            // add the packet to the message --same packet can be received more than once
                            //print_log_msg(".....................................................message " + pointer_to_string(message));
                            //print_log_msg(".....................................................message id    " + message->get_message_id());
                            //print_log_msg(".....................................................packet msg id " + packet->get_message_id());
                            if (! message->was_packet_already_added(packet->get_seq_no())) {
                                message->add_packet(packet);
                                // acknowledge the received packet
                            }
                            //  this->send_ack(packet->create_acknowledgement(distributed_environment.get_my_s_addr()));
                            //print_log_msg(".....................................................added packet to message and sent ack");
                        }
                        // if message complete enqueue the message
                        if (message->is_complete()) {
                            this->active_messages.erase(message->get_message_id());
                            if (message->is_checksum_valid()) {
                                switch (message->get_type()) {
                                    case eddl_message_types::DATA_SAMPLES:
                                    case eddl_message_types::DATA_GRADIENTS:
                                    case eddl_message_types::DATA_WEIGHTS:
                                        this->input_queue.push(message);
                                        this->recently_received_messages[message->get_message_id()] = get_system_milliseconds();
                                        this->output_queue.push_front(message->create_acknowledgement());
                                        print_log_msg(std::string("received message ") + message->get_message_id());
                                        break;
                                    default:
                                        throw std::runtime_error(err_msg("unexpected message type."));
                                }
                            } else {
                                delete message;
                            }
                        }
                    }
                    break;
                case eddl_message_types::COMMAND:
                    if (packet->get_command() == eddl_command_types::SHUTDOWN)
                        receiver_active = false;
                    this->input_queue.push(new eddl_message(packet));
                    break;
                case eddl_message_types::PARAMETER:
                    this->input_queue.push(new eddl_message(packet));
                    break;
                case eddl_message_types::PKG_ACK:
                case eddl_message_types::MSG_ACK_SAMPLES:
                case eddl_message_types::MSG_ACK_WEIGHTS:
                case eddl_message_types::MSG_ACK_GRADIENTS:
                    //this->ack_queue.push(new eddl_message(packet));
                    //break;
                default:
                    throw std::runtime_error(err_msg("unexpected message type."));
            } // switch
        } else {
            // otherwise do nothing, sender will resend non-acknowledged packets
            // so the next print_err_msg() must be commented
            print_err_msg("received packet " + std::to_string(packet->get_seq_no())
                        + "/" + std::to_string(packet->get_seq_len())
                        + " of message " + packet->get_message_id()
                        + " from " + get_ip_address(packet->get_source_addr()));
        }
        /*
            instead of deleting the object of the class eddl_packet, we have
            to free the memory block
            delete packet -- DON'T DO THIS IN THIS CASE
        */
        free(data);
    } // while receiver_active
    close(socket_fd_in);
    print_log_msg("multicast receiver thread stopped normally");
}
};
