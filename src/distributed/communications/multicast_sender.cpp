/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: August 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#include <eddl/distributed/multicast_sender.h>

#include <cstring>
#include <unistd.h>
#include <iostream>

namespace eddl {

MulticastSender::MulticastSender(std::vector<eddl_worker_node *> & workers,
                                 eddl_queue & output_queue,
                                 eddl_queue & ack_queue,
                                 DistributedEnvironment & distributed_environment) :
    workers(workers),
    output_queue(output_queue),
    ack_queue(ack_queue),
    distributed_environment(distributed_environment)
{
    socket_fd_out = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_fd_out < 0)
        throw std::runtime_error(err_msg("output socket cannot be created."));

    u_char loop = 0; // 0 to disable multicast loop ; not necessary in theory
                     // change this if the master als acts as a worker
    if (setsockopt(socket_fd_out, IPPROTO_IP, IP_MULTICAST_LOOP, &loop, sizeof(loop)) < 0)
        throw std::runtime_error(err_msg("cannot deactivate multicast loop."));

    u_char ttl=1; // set ttl to the number of routers multicast packets can go through
                  // change this to adapt to the needs of federated machine learning
    if (setsockopt(socket_fd_out, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)) < 0)
        throw std::runtime_error(err_msg("cannot set multicast TTL. " + std::to_string(errno) + ":" + strerror(errno)));

    memset(&this->target_group_addr, 0, sizeof(this->target_group_addr));
    this->target_group_addr.sin_family = AF_INET;
    this->target_group_addr.sin_addr.s_addr = distributed_environment.get_multicast_s_addr();
    this->target_group_addr.sin_port = htons(distributed_environment.get_udp_data_port());

    struct in_addr mreq;
    mreq.s_addr = distributed_environment.get_my_s_addr();
    /* alternative 1 sinc Linux 1.2
    struct ip_mreq mreq;
    mreq.imr_multiaddr.s_addr = 0; //distributed_environment.get_multicast_s_addr();
    mreq.imr_interface.s_addr = distributed_environment.get_my_s_addr();
    */
    /* alternative 2 since Linux 3.5
    struct ip_mreqn mreq;
    mreq.imr_multiaddr.s_addr = 0; //distributed_environment.get_multicast_s_addr();
    mreq.imr_address.s_addr = distributed_environment->get_my_s_addr();
    mreq.imr_ifindex = 0;
    */
    if (setsockopt(socket_fd_out, IPPROTO_IP, IP_MULTICAST_IF, (char *)&mreq, sizeof(mreq)) < 0)
        throw std::runtime_error(err_msg("cannot set my inferface addr for multicast."
                                        + std::to_string(errno) + ":" + strerror(errno)));

    ////////////////////////////////////////////////////////////////////////////
    socket_fd_in = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_in < 0)
        throw std::runtime_error(err_msg("input socket cannot be created."));

    struct sockaddr_in  host_addr;
    memset(&host_addr, 0, sizeof(host_addr));
    host_addr.sin_family = AF_INET;
    host_addr.sin_addr.s_addr = INADDR_ANY; // distributed_environment->get_my_s_addr();
    host_addr.sin_port = htons(distributed_environment.get_udp_ack_port());

    if (bind(socket_fd_in, (struct sockaddr *) &host_addr, sizeof(host_addr)) < 0)
        throw std::runtime_error(err_msg("binding socket failed."));

    ////////////////////////////////////////////////////////////////////////////

    std::cout   << "ready to sent messages to multicast group "
                << get_ip_address(distributed_environment.get_multicast_s_addr())
                << ":" << distributed_environment.get_udp_data_port()
                << " via " << get_ip_address(distributed_environment.get_my_s_addr())
                << " and receive acknowledgements from any worker via port "
                << distributed_environment.get_udp_ack_port()
                << std::endl;

    socklen_t   optlen;
    int         sockt_buffer_size;
    int         rc;
/*
    optlen = sizeof(sockt_buffer_size);
    sockt_buffer_size = 30*1024*1024;
    rc = setsockopt(socket_fd_out, SOL_SOCKET, SO_SNDBUF, &sockt_buffer_size, optlen);
    std::cout << "rc = " << rc << std::endl;
*/
    optlen = sizeof(sockt_buffer_size);
    rc = getsockopt(socket_fd_out, SOL_SOCKET, SO_SNDBUF, &sockt_buffer_size, &optlen);
    std::cout   << "send UDP buffer size is " << sockt_buffer_size
                << "  rc = " << rc
                << std::endl;
    optlen = sizeof(sockt_buffer_size);
    rc = getsockopt(socket_fd_out, SOL_SOCKET, SO_RCVBUF, &sockt_buffer_size, &optlen);
    std::cout   << "recv UDP buffer size is " << sockt_buffer_size
                << "  rc = " << rc
                << std::endl;
    ////////////////////////////////////////////////////////////////////////////

    sender_active = true;
    sender_thread = std::thread( & MulticastSender::sender, this);
    ack_processor_thread = std::thread( & MulticastSender::ack_processor, this);
}

MulticastSender::~MulticastSender()
{
    stop();

    sender_active = false;
    output_queue.clear();
    output_queue.push(nullptr);

    sender_thread.join();
    ack_processor_thread.join();

    for (auto iter: this->active_acknowledgements)
        delete iter.second;
}

void MulticastSender::stop()
{
    sender_active = false;
    ////////////////////////////////////////////////////////////////////////////
    ////////// this stops the acknowledgement procesor thread //////////////////
    int temp_socket = socket(AF_INET, SOCK_DGRAM, 0);
    char data[sizeof(eddl_packet_ack)];
    memset(data, 0, sizeof(data));
    struct sockaddr_in peer;
    memset(&peer, 0, sizeof(peer));
    peer.sin_family = AF_INET;
    peer.sin_port = htons(distributed_environment.get_udp_ack_port());
    peer.sin_addr.s_addr = distributed_environment.get_my_s_addr();
    ssize_t l = sizeof(data);
    ssize_t n = sendto(temp_socket, data, l, MSG_NOSIGNAL,
                        (const struct sockaddr *)&peer, sizeof(peer));
    if (n != l)
        print_err_msg("failed to sent a stopping acknowledgement to myself.");
    close(temp_socket);
    ////////////////////////////////////////////////////////////////////////////
}

void MulticastSender::sender()
{
    eddl_message * message = nullptr;
    while (sender_active) {
        if (output_queue.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        // poping from the queue blocks until something his available
        message = output_queue.pop();
        // this allows to stop this thread, according to the destructor of this class
        if (nullptr == message) {
            continue;
        } else if (! send_message(message)) {
            // an error ocurred while sending the message
            std::string msg_id = message->get_message_id();
            // destroys the message
            print_err_msg("message " + msg_id + " was not sent to all living workers.");
        }
        delete message;
    }
    close(socket_fd_out);
    print_log_msg("multicast sender thread stopped normally.");
}
void MulticastSender::ack_processor()
{
    int                 flags = MSG_NOSIGNAL;
    struct sockaddr_in  peer_addr;
    socklen_t           peer_addr_size;
    unsigned char       data[next_multiple(sizeof(eddl_packet_ack),8)];

    while (sender_active) {
        ssize_t l = sizeof(eddl_packet_ack);
        memset(data, 0, sizeof(data));
        memset(&peer_addr, 0, sizeof(peer_addr));
        peer_addr_size=0;
        // blocking call
        ssize_t n = recvfrom(socket_fd_in, data, l, flags, (struct sockaddr *)&peer_addr, &peer_addr_size);
        if (n < 0) {
            print_err_msg("error receiving an acknowledgement: "
                            + std::to_string(errno) + ": " + strerror(errno));
            continue; // do not abort the process, just drop the packet
        }
        if (n != l) {
            print_err_msg("warning received an incomplete acknowledgement of "
                        + std::to_string(n) + " bytes instead of "
                        + std::to_string(l) + " bytes requested");
            continue; // do not abort the process, just drop the packet
        }

        eddl_packet_ack * ack = (eddl_packet_ack *)data;

        ////////////////////////////////////////////////////////////////////////
        size_t sum=0;
        for (size_t i = 0; i < sizeof(eddl_packet_ack); i++) sum += data[i];
        if (0 == sum) break; // an empty acknowledgement means to stop
        ////////////////////////////////////////////////////////////////////////

        std::string message_id = ack->get_message_id();
        { // critical region starts
            std::unique_lock<std::mutex> lck(ack_processor_mutex);

            if (this->active_acknowledgements.count(message_id) > 0) {
                eddl_message_acks * _acks = this->active_acknowledgements[message_id];
                _acks->acknowledge(ack->get_source_addr(), ack->get_seq_no());
                if (distributed_environment.get_verbose_level() > 2)
                    print_log_msg("received acknowledgement "
                                    + std::to_string(ack->get_seq_no())
                                    + " for message " + message_id);
            } else {
                print_log_msg("received an obsolete acknowledgement for message " + message_id);
            }
        } // critical region ends
    }
    close(socket_fd_in);
    print_log_msg("multicast acknowledgment processor thread stopped normally.");
}

bool MulticastSender::send_message(eddl_message * message)
{
    message->set_source_addr(distributed_environment.get_my_s_addr());
    message->set_target_addr(distributed_environment.get_multicast_s_addr());
    // compulsory to compute again the message id every time source addr is updated
    message->set_message_id(); // with no parameter method set_message_id() computes the message_id

    eddl_message_acks * message_acks = nullptr;

    // prepare acknowledgements for the message to be sent here
    { // critical region starts
        std::unique_lock<std::mutex> lck(ack_processor_mutex);

        message_acks = new eddl_message_acks(workers, message);
        this->active_acknowledgements[message->get_message_id()] = message_acks;
    } // critical region ends

    std::queue<size_t>  seq_no_queue;
    eddl_packet * sent_packets[message->get_seq_len()+1]; // includes the checksum

    // populates the queue of packet indices including one additional for the checksum
    for (size_t seq_no=0; seq_no <= message->get_seq_len(); ++seq_no) {
        seq_no_queue.push(seq_no);
        sent_packets[seq_no] = nullptr;
    }

    bool return_status = true;
    int flags = MSG_NOSIGNAL;

    try {
        uint64_t    t0 = get_system_milliseconds();
        size_t      msec_to_wait_after_sendto=1;
        size_t      counter=0;
        ssize_t     sent_bytes=0;
        while( sender_active  &&  ! seq_no_queue.empty()) {
            size_t  pending_packets = seq_no_queue.size();
            for (size_t i=0; sender_active && i < pending_packets; i++) {
                size_t seq_no = seq_no_queue.front();
                seq_no_queue.pop();

                bool packet_to_be_sent = false;
                { // critical region starts
                    std::unique_lock<std::mutex> lck(ack_processor_mutex);

                    packet_to_be_sent = ! message_acks->packet_already_acknowledged(seq_no);
                } // critical region ends

                if (packet_to_be_sent) {
                    eddl_packet * packet = sent_packets[seq_no];
                    if (nullptr == packet) {
                        if (seq_no < message->get_seq_len())
                            packet = message->get_packet(seq_no);
                        else
                            packet = message->create_packet_for_checksum();
                        sent_packets[seq_no] = packet;
                    }
                    ssize_t l = sizeof(eddl_packet);
                    ssize_t n = sendto(socket_fd_out, (void *)packet, l,
                                       flags,
                                       (const struct sockaddr *) &this->target_group_addr,
                                       sizeof(this->target_group_addr));
                    sent_bytes += n;
                    if (sent_bytes >= 4000) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(msec_to_wait_after_sendto));
                        sent_bytes -= 4000;
                    }
                    if (n != l)
                        throw std::runtime_error(err_msg("sent " + std::to_string(n)
                                                       + " bytes instead of " + std::to_string(l)
                                                       + "  " + std::to_string(errno) + ": "
                                                       + strerror(errno)));
                    //print_log_msg("packet sent " + std::to_string(seq_no));
                    seq_no_queue.push(seq_no);
                } else if (nullptr != sent_packets[seq_no]) {
                    delete sent_packets[seq_no];
                    sent_packets[seq_no] = nullptr;
                }
                // otherwise the packet at seq_no is considered successfully sent
                // and is not pushed into the queue again

                if (message_acks->lasting_too_much_time()) {
                    return_status = false;
                    throw std::runtime_error(err_msg("time over sending message " + message->get_message_id()));
                }

                ++counter;
            } // inner for loop  i < pending_packets

            if (! ack_queue.empty()) {
                eddl_message * ack = ack_queue.pop();

                std::string message_id = ack->get_acknowledged_message_id();
                if (ack->get_type() == eddl_message_types::MSG_ACK_WEIGHTS) {
                // critical region starts
                    std::unique_lock<std::mutex> lck(ack_processor_mutex);

                    if (this->active_acknowledgements.count(message_id) > 0) {
                        eddl_message_acks * _acks = this->active_acknowledgements[message_id];
                        _acks->acknowledge_whole_message(ack->get_source_addr());
                        if (distributed_environment.get_verbose_level() > 1)
                            print_log_msg("received acknowledgement of whole message:" + message_id);
                    } else {
                        if (distributed_environment.get_verbose_level() > 1)
                            print_err_msg("received acknowledgement of a non-active whole message:" + message_id);
                    }
                } // critical region ends
                delete ack;
            }

            print_log_msg("message being sent: " + message->get_message_id()
                            + "  |seq_no_queue| = " + std::to_string(seq_no_queue.size())
                            + "  pending_acknowledgement_count = "
                            + std::to_string(message_acks->get_pending_acknowledgements())
                            + "  counter = " + std::to_string(counter)
                            + "  waiting " + std::to_string((get_system_milliseconds()-t0))
                            + " milliseconds from message started to be sent.");
            /*
            msec_to_wait_after_sendto = (msec_to_wait_after_sendto == 1)
                                      ? 10
                                      : msec_to_wait_after_sendto+10;
            */
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } // outer while loop ! seq_no_queue.empty()
    }
    catch(std::exception & e) {
        print_err_msg(std::string("an exception ocurred: ") + e.what());
        return_status = false;
    }

    // cleaning data structures starts
    while (! seq_no_queue.empty()) seq_no_queue.pop();

    for (auto packet : sent_packets)
        if (nullptr != packet) delete packet;
    // cleaning data structures ends

    { // critical region starts
        std::unique_lock<std::mutex> lck(ack_processor_mutex);

        if (message_acks->all_has_been_acknowledged()) {
            // do any pending action to do
        } else {
            // review the list of workers and deactivate those who failed systematically
            return_status = false;
        }
        this->active_acknowledgements.erase(message->get_message_id());
        delete message_acks;
    } // critical region ends

    return return_status;
}

}; // namespace eddl
