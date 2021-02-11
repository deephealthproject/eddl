
#include <eddl/distributed/eddl_distributed.h>
#include <eddl/distributed/eddl_worker_node.h>
#include <eddl/distributed/eddl_packet.h>
#include <eddl/distributed/eddl_message.h>

#include <map>
#include <iostream>

using namespace eddl;

int main(int argc, char *argv[])
{
    std::cout
        << "  sizeof(eddl_packet)   = " << sizeof(eddl_packet)      << std::endl
        << "  sizeof(eddl_message)  = " << sizeof(eddl_message)     << std::endl
        << "  eddl_default_mtu      = " << eddl_default_mtu         << std::endl
        << "  eddl_packet_data_size = " << eddl_packet_data_size    << std::endl
        << "  eddl_checksum_len     = " << eddl_checksum_len        << std::endl
        << "  eddl_msg_id_len       = " << eddl_msg_id_len          << std::endl
        << " _eddl_msg_id_len_      = " << _eddl_msg_id_len_        << std::endl
        << "  eddl_alignment        = " << eddl_alignment           << std::endl
        << "  listen_max_pending    = " << listen_max_pending       << std::endl
        << "  base_tcp_port         = " << base_tcp_port            << std::endl
        << "  base_udp_data_port    = " << base_udp_data_port       << std::endl
        << "  base_udp_ack_port     = " << base_udp_ack_port        << std::endl
        << std::endl;

    char buffer[1024*1024];
    eddl_message * message = new eddl_message(DATA_GRADIENTS, 0, 0, 1024*1024, eddl_packet_data_size, buffer);

    std::cout
        << "  message->get_message_data_size()    = " << message->get_message_data_size()       << std::endl
        << "  message->get_seq_len()              = " << message->get_seq_len()                 << std::endl
        << std::endl;

    eddl_packet * packet = message->get_packet(0);

    std::cout
        << "  packet->get_message_size()          = " << packet->get_message_size()       << std::endl
        << "  packet->get_data_size()             = " << packet->get_data_size()          << std::endl
        << std::endl;

    delete packet;
    delete message;

    init_message_type_names();
    show_all_message_type_names();

    return EXIT_SUCCESS;
}
