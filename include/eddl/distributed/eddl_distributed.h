/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: x.y
 * copyright (c) 2020, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: July 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#ifndef __EDDL_DISTRIBUTED_H__
#define __EDDL_DISTRIBUTED_H__ 1

#include <string>
#include <vector>

namespace eddl {

enum eddl_thread_status {INACTIVE, RUNNING, STOPPED};
enum eddl_worker_status {   WORKER_WAITING,
                            WORKER_RUNNING,
                            WORKER_STOPPING,
                            WORKER_TO_SHUTDOWN};
/*
    - a worker starts in the WAITING state
    - and can move
        + from WAITING to RUNNING by means of a command from the master
        + from WAITING to TO_SHUTDOWN by means of a command from the master
        + from RUNNING to STOPPING by means of a command from the master
        + from STOPPING to WAITING after completing and/or aborting pending tasks
        + from STOPPING to TO_SHUTDOWN by means of a command from the master

    - a worker can be mastered by a master if configure in modes FEDERATED_ML or ANY_MASTER
        + the master sends the command MASTER_A_WORKER to a worker to master it
        + a worker accepts if it was free, or rejects the command
        + the master sends the command FREE_A_WORKER to a worker to free it, then the worker will complete any pending task
        + a worker becomes free if no data nor commands from master are received during a time lapse to be adjusted

    - when a worker ends its execution, a new worker process can be launched
      automatically if the system is configured to do it, otherwise it must be
      launched manually.
*/

enum eddl_message_types {DATA_WEIGHTS          = 0x00501,
                         DATA_GRADIENTS        = 0x00502,
                         DATA_SAMPLES          = 0x00504,

                         PARAMETER             = 0x00a01,
                         COMMAND               = 0x00a02,
                         PKG_ACK               = 0x00a04,
                         MSG_CHKSUM            = 0x00a08,
//                         PKG_CHKSUM          = 0x00a10,
                         MSG_ACK_WEIGHTS       = 0x00a21,
                         MSG_ACK_GRADIENTS     = 0x00a22,
                         MSG_ACK_SAMPLES       = 0x00a24,

                         MASTER_A_WORKER       = 0x00011,
                         FREE_A_WORKER         = 0x00012,
                         ACCEPT_TO_BE_MASTERED = 0x00014,
                         REJECT_TO_BE_MASTERED = 0x00018
                         };

enum eddl_command_types {START    = 0x041,
                         STOP     = 0x042,
                         SHUTDOWN = 0x044};

enum eddl_worker_modes {FEDERATED_ML = 0x011, // no data is accepted from the master
                        ONE_MASTER =   0x022, // only obey to one master that must be specified
                        ANY_MASTER =   0x044  // worker servers to any master if not busy
                        };

static constexpr int base_tcp_port      = 3017; ///< port master node will accept connections from worker nodes: 3x17, where x={0..9}
static constexpr int base_udp_data_port = 3011; ///< port master node will send datagrams to worker nodes
static constexpr int base_udp_ack_port  = 3013; ///< port master node will receive acknowledgements from worker nodes

// see https://www.cisco.com/c/dam/en/us/support/docs/ip/ip-multicast/ipmlt_wp.pdf
static std::string  eddl_multicast_group_addr("239.193.111.211"); // campus scope
//static std::string  eddl_multicast_group_addr("225.1.1.1"); // testing example

#define next_multiple(_x_,_y_)  ((_y_)*(((_x_)/(_y_))+(((_x_)%(_y_))!=0)))
#define prev_multiple(_x_,_y_)  ((_y_)*(((_x_)/(_y_))))

static constexpr size_t eddl_alignment = 8; ///< alignment in bytes to allocate memory
static constexpr int listen_max_pending = 50; ///< maximum number of connections pending to be accepted by the master node
static constexpr int eddl_checksum_len = 32; ///< SHA256 algorithm is used, whose output is 256 bits (32 bytes) length
static constexpr size_t eddl_msg_id_len = 19; ///< 19=8+3+8 hexadecimal digits, 8 of the IP address, 3 of the message type and 8 of the timestamp in milliseconds
static constexpr size_t _eddl_msg_id_len_ = next_multiple(eddl_msg_id_len,eddl_alignment); ///< next eight-multiple from 19
static constexpr size_t eddl_default_mtu = 8192; // 1500; //1536; ///< MTU -- block size for sending/receiving packets (mainly affects UDP multicast)
static constexpr size_t eddl_packet_data_size = prev_multiple(eddl_default_mtu
                                                - 4*sizeof(uint32_t)
                                                - _eddl_msg_id_len_
                                                - 4*sizeof(size_t)
                                                - eddl_checksum_len, 8);
                                                // check this with eddl_packet class definition

uint64_t                    get_system_milliseconds();
std::vector<std::string>    str_split(std::string s, char sep);
std::string                 get_ip_address(uint32_t s_addr);
std::string                 pointer_to_string(void * ptr);

size_t compute_aligned_size(size_t size);
void * eddl_malloc(size_t size);

std::string compose_log_message(const char * filename, const int line_number, const char * function_name, const char *msg);
std::string compose_log_message(const char * filename, const int line_number, const char * function_name, std::string msg);
#define err_msg(s)  compose_log_message(__FILE__,__LINE__,__func__,s)
void print_log_message(const char * filename, const int line_number, const char * function_name, const char *msg);
void print_log_message(const char * filename, const int line_number, const char * function_name, std::string msg);
#define print_log_msg(s)  print_log_message(__FILE__,__LINE__,__func__,s)
void print_err_message(const char * filename, const int line_number, const char * function_name, const char *msg);
void print_err_message(const char * filename, const int line_number, const char * function_name, std::string msg);
#define print_err_msg(s)  print_err_message(__FILE__,__LINE__,__func__,s)

void init_message_type_names();
std::string get_message_type_name(int value);
void show_all_message_type_names();

};

#endif // __EDDL_DISTRIBUTED_H__
