#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>
#include <random>
#include <iostream>

#include <eddl/distributed/distributed_environment.h>
#include <eddl/distributed/tcp_receiver.h>
#include <eddl/distributed/tcp_sender.h>
#include <eddl/distributed/multicast_receiver.h>

int main(int argc, char *argv[])
{
    eddl::DistributedEnvironment    distributed_environment;
    eddl::eddl_worker_modes         worker_mode = eddl::eddl_worker_modes::ANY_MASTER;

    // manual settings for testing
    //distributed_environment.set_my_ip_addr("10.81.25.1"); // platon.vpn
    //distributed_environment.set_master_ip_addr("10.81.25.6"); // socrates.vpn
    distributed_environment.set_my_ip_addr("158.42.215.16");  //ebids.etsinf
    distributed_environment.set_master_ip_addr("158.42.184.139"); // platon.dsic

    for (int i = 0; i < argc; i++) {
        if (! strcmp(argv[i], "--my-ip-addr")) {
            distributed_environment.set_my_ip_addr(argv[++i]);
        } else if (! strcmp(argv[i], "--server")) {
            distributed_environment.set_master_ip_addr(argv[++i]);
        } else if (! strcmp(argv[i], "--tcp-port")) {
            distributed_environment.set_tcp_port(atoi(argv[++i]));
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

        } else if (! strcmp(argv[i], "--multicast-group-addr")) {
            distributed_environment.set_multicast_group_addr(argv[++i]);
        } else if (! strncmp(argv[i], "--verbose=", 10)) {
            std::vector<std::string> parts = eddl::str_split(argv[i],'=');
            distributed_environment.set_verbose_level(std::stoi(parts[1]));
        } else if (! strcmp(argv[i], "--verbose")) {
            distributed_environment.increase_verbose_level();
        }
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 generator_1(seed);      // mt19937 is a standard mersenne_twister_engine
    std::mt19937_64 generator_2(seed + 3);  // mt19937 is a standard mersenne_twister_engine
    std::uniform_int_distribution<int> dist_sizes(100,50*1024*1024);
    std::uniform_int_distribution<int> dist_content(0,255);

    eddl::eddl_queue    input_queue;
    eddl::eddl_queue    output_queue;
    eddl::eddl_queue    ack_queue;

    eddl::TCP_Receiver  tcp_receiver(   input_queue,
                                        ack_queue,
                                        ack_queue,
                                        output_queue,
                                        distributed_environment);
    eddl::TCP_Sender    tcp_sender(     output_queue,
                                        ack_queue,
                                        distributed_environment);
    eddl::MulticastReceiver multicast_receiver( input_queue,
                                                ack_queue,
                                                output_queue,
                                                distributed_environment);

    eddl::eddl_message *        message = nullptr;
    eddl::eddl_worker_status    worker_status = eddl::eddl_worker_status::WORKER_WAITING;
    int seconds_to_wait_while_waiting = 1;
    int iterations_waiting = 0;

    while (worker_status != eddl::eddl_worker_status::WORKER_TO_SHUTDOWN) {

        // independently of the status the input queue must be processed
        //if
        while (! input_queue.empty()) {
            message = input_queue.pop();
            std::cout << "received message: "
                      << std::hex << message->get_message_id() << " "
                      << std::hex << message->get_type() << " "
                      << std::dec << message->get_timestamp() << " "
                      << std::dec << message->get_message_data_size()
                      << " bytes" << std::endl;

            switch (message->get_type()) {
                case eddl::eddl_message_types::COMMAND:
                    switch (message->get_command()) {
                        case eddl::eddl_command_types::START:
                            if (worker_status == eddl::eddl_worker_status::WORKER_WAITING) {
                                worker_status = eddl::eddl_worker_status::WORKER_RUNNING;
                                seconds_to_wait_while_waiting = 1;
                                iterations_waiting = 0;
                            }
                            break;
                        case eddl::eddl_command_types::STOP:
                            switch (worker_status) {
                                case eddl::eddl_worker_status::WORKER_WAITING:
                                case eddl::eddl_worker_status::WORKER_RUNNING:
                                    worker_status = eddl::eddl_worker_status::WORKER_STOPPING;
                                    break;
                                default:
                                    break;
                            }
                            break;
                        case eddl::eddl_command_types::SHUTDOWN:
                            worker_status = eddl::eddl_worker_status::WORKER_TO_SHUTDOWN;
                            break;
                    }
                    break;

                case eddl::eddl_message_types::MASTER_A_WORKER:
                    output_queue.push(new eddl::eddl_message(distributed_environment.is_master_ip_addr_set()
                                                                ? eddl::eddl_message_types::REJECT_TO_BE_MASTERED
                                                                : eddl::eddl_message_types::ACCEPT_TO_BE_MASTERED,
                                                             0, // source addr will be set by the sender thread
                                                             message->get_source_addr(),
                                                             0,
                                                             eddl::eddl_packet_data_size,
                                                             nullptr));

                    if (! distributed_environment.is_master_ip_addr_set()) {
                        distributed_environment.set_master_s_addr(message->get_source_addr());
                    }
                    break;

                case eddl::eddl_message_types::FREE_A_WORKER:
                    distributed_environment.clear_master_ip_addr();
                    switch (worker_status) {
                        case eddl::eddl_worker_status::WORKER_WAITING:
                        case eddl::eddl_worker_status::WORKER_RUNNING:
                            worker_status = eddl::eddl_worker_status::WORKER_STOPPING;
                            break;
                        default:
                            break;
                    }
                    break;
            }
            delete message;
        }

        switch (worker_status) {
            case eddl::eddl_worker_status::WORKER_RUNNING:
                if (output_queue.size() < 10) {
                    size_t size_in_bytes = dist_sizes(generator_1);
                    unsigned char * data = new unsigned char [size_in_bytes];
                    for (size_t i=0; i < size_in_bytes; i++)
                        data[i] = (unsigned char)dist_content(generator_1);

                    message = new eddl::eddl_message(
                                    eddl::eddl_message_types::DATA_GRADIENTS,
                                    0, // source addr will be set by the sender thread
                                    distributed_environment.get_master_s_addr(),
                                    size_in_bytes,
                                    eddl::eddl_packet_data_size,
                                    data);
                    delete [] data;
                    output_queue.push(message);
                } else {
                    /*
                        sleep 1 second to avoid having too many messages in the
                        output queue; this should be reviewed in the real
                        worker implementation
                    */
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
                break;

            case eddl::eddl_worker_status::WORKER_STOPPING:
                // does not perform new actions like sending new messages
                if (output_queue.empty())
                    worker_status = eddl::eddl_worker_status::WORKER_WAITING;
                else
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                break;

            case eddl::eddl_worker_status::WORKER_TO_SHUTDOWN:
                output_queue.clear();
                tcp_sender.stop();
                tcp_receiver.stop();
                multicast_receiver.stop();
                break;

            case eddl::eddl_worker_status::WORKER_WAITING:
                std::cout << "worker inactive waiting for "
                          << seconds_to_wait_while_waiting
                          << " second(s)." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(seconds_to_wait_while_waiting));
                if (++iterations_waiting >= 10)
                    seconds_to_wait_while_waiting = 10;
                break;
        } // of switch

        if (distributed_environment.get_verbose_level() >= 1)
            std::cout << "  |input_queue| = " << input_queue.size()
                      << "  |output_queue| = " << output_queue.size()
                      << "  |ack_queue| = " << ack_queue.size()
                      << std::endl;
    } // of while worker_status

    eddl::print_log_msg("worker main thread ready to finish when threads stop");

    return EXIT_SUCCESS;
}
