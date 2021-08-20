#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <cstring>
#include <csignal>
#include <chrono>
#include <random>
#include <iostream>

#include <eddl/distributed/tcp_receiver.h>
#include <eddl/distributed/tcp_sender.h>
#include <eddl/distributed/multicast_sender.h>
#include <eddl/distributed/eddl_worker_node.h>

eddl::eddl_queue * global_input_queue;

void handler_funtion(int parameter)
{
    static int counter = 0;

    switch (parameter) {
        case SIGUSR1:
            global_input_queue->push(eddl::eddl_message::start_command(0));
            break;
        case SIGHUP:
            global_input_queue->push(eddl::eddl_message::stop_command(0));
            break;
        case SIGINT: // also generated from keyboard by CTRL+C in Unix systems
        case SIGUSR2:
        case SIGTERM:
            global_input_queue->push(eddl::eddl_message::shutdown_command(0));
            break;
    }
    eddl::print_log_msg("signal caught " + std::to_string(parameter));
    if (++counter > 10) exit(1);
}

int main(int argc, char *argv[])
{
    eddl::DistributedEnvironment distributed_environment;
    //distributed_environment.set_my_ip_addr("10.81.25.6"); // socrates.vpn
    distributed_environment.set_my_ip_addr("158.42.184.139"); // platon.dsic
    eddl::eddl_queue    input_queue;
    eddl::eddl_queue    generic_output_queue;
    eddl::eddl_queue    generic_ack_queue;
    eddl::eddl_queue    weights_output_queue;
    eddl::eddl_queue    weights_ack_queue;
    std::vector<eddl::eddl_worker_node *> workers;

    for (int i=0; i < argc; i++) {
        if (! strcmp(argv[i], "--my-ip-addr")) {
            distributed_environment.set_my_ip_addr(argv[++i]);
        } else if (! strcmp(argv[i], "--tcp-port")) {
            distributed_environment.set_tcp_port(atoi(argv[++i]));
        /*
        } else if (! strncmp(argv[i], "--mode=", 7)) {
            std::vector<std::string> parts = eddl::str_split(argv[i],'=');
            if (parts[1] == "federated_ml")
                worker_mode = eddl::eddl_worker_modes::FEDERATED_ML;
            else if (parts[1] == "one_master")
                worker_mode = eddl::eddl_worker_modes::ONE_MASTER;
            else if (parts[1] == "any_master")
                worker_mode = eddl::eddl_worker_modes::ANY_MASTER;
            else
                throw std::runtime_error(eddl::err_msg("unrecognized worker mode"));
        */
        } else if (! strcmp(argv[i], "--multicast-group-addr")) {
            distributed_environment.set_multicast_group_addr(argv[++i]);
        } else if (! strncmp(argv[i], "--verbose=", 10)) {
            std::vector<std::string> parts = eddl::str_split(argv[i],'=');
            distributed_environment.set_verbose_level(std::stoi(parts[1]));
        } else if (! strcmp(argv[i], "--verbose")) {
            distributed_environment.increase_verbose_level();
        }
    }

    //workers.push_back(new eddl::eddl_worker_node("ip:10.81.25.1;cpu:4,8192;gpu:0,low_mem;fpga:0,0;batch_size:10"));
    //workers.push_back(new eddl::eddl_worker_node("ip:158.42.215.16;cpu:8,131072;gpu:0,low_mem;fpga:0,0;batch_size:10"));
    workers.push_back(new eddl::eddl_worker_node("ip:192.168.1.1;cpu:8,131072;gpu:0,low_mem;fpga:0,0;batch_size:10"));

    /*
        tcp_receiver pushes:
            1. weights acknowledgements from workers into the weights_ack_queue
            2. other acknowledgements from workers into the generic_ack_queue
            3. acknowledgements created by the master when a data message is complete into the generic_output_queue
            4. messages which are not acknowledgements into the input_queue
    */
    eddl::TCP_Receiver  tcp_receiver(   input_queue,
                                        weights_ack_queue,
                                        generic_ack_queue,
                                        generic_output_queue,
                                        distributed_environment);
    /*
        tcp_sender pops:
            1. output messages from the generic_output_queue
            2. other acknowledgements sent by workers from the generic_ack_queue

    */
    eddl::TCP_Sender    tcp_sender( generic_output_queue,
                                    generic_ack_queue,
                                    distributed_environment);
    /*
        multicast_sender pops:
            1. output weight data messages from the weights_output_queue
            2. weights acknowledgements sent by workers from the weights_ack_queue
    */
    eddl::MulticastSender    multicast_sender(  workers,
                                                weights_output_queue,
                                                weights_ack_queue,
                                                distributed_environment);

    ////////////////////////////////////////////////////////////////
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 generator_1(seed);      // mt19937 is a standard mersenne_twister_engine
    std::mt19937_64 generator_2(seed + 1);  // mt19937 is a standard mersenne_twister_engine
    std::mt19937_64 generator_3(seed + 2);  // mt19937 is a standard mersenne_twister_engine
    std::bernoulli_distribution bernoulli(0.3);
    std::uniform_int_distribution<int> dist_sizes(100, 50 * 1024 * 1024);
    std::uniform_int_distribution<int> dist_content(0, 255);
    ////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////
    global_input_queue = & input_queue;
    // setting sigaction structure: begin
    static struct sigaction _action;
    _action.sa_sigaction = nullptr;
    _action.sa_handler = handler_funtion;
    for (unsigned long int i=0; i < _SIGSET_NWORDS; i++)
        _action.sa_mask.__val[i] = 0;
    _action.sa_flags = 0;
    // setting sigaction structure: end
    sigaction(SIGHUP,  &_action, nullptr);
    sigaction(SIGINT,  &_action, nullptr);
    sigaction(SIGTERM, &_action, nullptr);
    sigaction(SIGUSR1, &_action, nullptr);
    sigaction(SIGUSR2, &_action, nullptr);
    ////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////
    eddl::print_log_msg("ready to receive messages");

    bool master_active = true;
    bool shutdown_master = false;

    for (auto w: workers)
        generic_output_queue.push(eddl::eddl_message::start_command(w->get_s_addr()));

    while (! shutdown_master) {
        eddl::eddl_message * msg = input_queue.pop();
        if (nullptr == msg) continue;


        std::cout << "received message: "
                  << std::hex << msg->get_message_id() << " "
                  << std::hex << msg->get_type() << " "
                  << std::dec << msg->get_timestamp() << " "
                  << std::dec << msg->get_message_data_size()
                  << " bytes" << std::endl;

        if (msg->get_type() == eddl::eddl_message_types::COMMAND) {
            switch (msg->get_command()) {

                case eddl::eddl_command_types::START:
                    master_active = true;
                    for (auto w: workers)
                        generic_output_queue.push(eddl::eddl_message::start_command(w->get_s_addr()));
                    break;

                case eddl::eddl_command_types::STOP:
                    master_active = false;
                    for (auto w: workers)
                        generic_output_queue.push(eddl::eddl_message::stop_command(w->get_s_addr()));
                    break;

                case eddl::eddl_command_types::SHUTDOWN:
                    master_active = false;
                    for (auto w: workers)
                        generic_output_queue.push(eddl::eddl_message::stop_command(w->get_s_addr()));
                    shutdown_master = true;
                    generic_output_queue.clear();
                    weights_output_queue.clear();
                    // first send a shutdown command to ensure tcp receivers change status to stop
                    for (auto w: workers)
                        generic_output_queue.push(eddl::eddl_message::shutdown_command(w->get_s_addr()));
                    // a first packet to myself to make my tcp receiver be aware of stopping
                    generic_output_queue.push(eddl::eddl_message::shutdown_command(distributed_environment.get_my_s_addr()));
                    // a packet to unlock multicast receiver threads
                    weights_output_queue.push(eddl::eddl_message::shutdown_command(0));
                    // second send a shutdown command to unlock the acceptor thread of tcp receivers
                    for (auto w: workers)
                        generic_output_queue.push(eddl::eddl_message::shutdown_command(w->get_s_addr()));
                    // a second packet to myself to stop my tcp receiver
                    generic_output_queue.push(eddl::eddl_message::shutdown_command(distributed_environment.get_my_s_addr()));
                    // wait a little bit
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    tcp_sender.stop();
                    tcp_receiver.stop();
                    multicast_sender.stop();
                    break;
            }
        }

        delete msg;

        if (master_active && weights_output_queue.size() < 10 && bernoulli(generator_1)) {
            size_t size_in_bytes = dist_sizes(generator_2);
            unsigned char * data = new unsigned char [size_in_bytes];
            for (size_t i=0; i < size_in_bytes; i++) data[i] = (unsigned char)dist_content(generator_3);

            msg = new eddl::eddl_message(eddl::eddl_message_types::DATA_WEIGHTS,
                                         0, 0,
                                         size_in_bytes, eddl::eddl_packet_data_size, data);

            delete [] data;
            weights_output_queue.push(msg);
        }

        std::cout << "  |input_queue| = " << input_queue.size()
                  << "  |generic_output_queue| = " << generic_output_queue.size()
                  << "  |weights_output_queue| = " << weights_output_queue.size()
                  << "  |generic_ack_queue| = " << generic_ack_queue.size()
                  << "  |weights_ack_queue| = " << weights_ack_queue.size()
                  << std::endl;
    }

    for (auto w : workers) delete w;
    workers.clear();

    eddl::print_log_msg("master main thread ready to finish when threads stop");

    return 0;
}
