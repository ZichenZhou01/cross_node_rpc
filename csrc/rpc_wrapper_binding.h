#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
#include <atomic>
#include <string>
#include <signal.h>
#include <rdma/rdma_cma.h>
#include <memory>
#include <cstring>
#include "rpc_helper.h"


// Starts the RDMA-based RPC server on the given port (default 7471)
void start_server(const std::string& port = "7471");
void stop_server();

class RpcClientWrapper {
public:
    RpcClientWrapper(const std::string& host, const std::string& port = "7471");
    std::string echo(const std::string& msg);
    int add(int a, int b);
    int mul(int a, int b, int c);
    uint64_t exchange_gpu_data(uint64_t send_gpu_addr, uint64_t send_size, 
        uint64_t recv_size, uint64_t recv_gpu_addr = 0);
    ~RpcClientWrapper();

private:
    rdma_event_channel* ec;
    rdma_cm_id* cmId;
    std::unique_ptr<RdmaTransport> transport;
    std::unique_ptr<RpcService> rpc;
    uint32_t request_counter_;
};