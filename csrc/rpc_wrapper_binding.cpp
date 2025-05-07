#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
#include <string>
#include <netdb.h>
#include <signal.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include "rpc_wrapper_binding.h"

namespace py = pybind11;

static std::thread server_thread;
static std::atomic<bool> server_running{false};

void start_server(const std::string& port) {
    if (server_running.load())
        throw std::runtime_error("Server already running");
    server_running = true;
    server_thread = std::thread([port](){
        try {
            int ret = run_server(port.c_str());
            if (ret != 0)
                std::fprintf(stderr, "Server exited with error code %d\n", ret);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "Server thread exception: %s\n", ex.what());
        }
    });
    server_thread.detach();
}

void stop_server() {
    stop_flag = true;
    if (server_thread.joinable())
        server_thread.join();
}

RpcClientWrapper::RpcClientWrapper(const std::string& host, const std::string& port) {
    signal(SIGINT, signal_handler);

    ec = rdma_create_event_channel();
    if (!ec) throw std::runtime_error("rdma_create_event_channel failed");

    if (rdma_create_id(ec, &cmId, nullptr, RDMA_PS_TCP))
        throw std::runtime_error("rdma_create_id failed");

    struct addrinfo hints = {};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* res;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res))
        throw std::runtime_error("getaddrinfo failed");

    if (rdma_resolve_addr(cmId, nullptr, res->ai_addr, 2000))
        throw std::runtime_error("rdma_resolve_addr failed");
    freeaddrinfo(res);

    wait_event(ec, cmId, RDMA_CM_EVENT_ADDR_RESOLVED);
    if (rdma_resolve_route(cmId, 2000))
        throw std::runtime_error("rdma_resolve_route failed");
    wait_event(ec, cmId, RDMA_CM_EVENT_ROUTE_RESOLVED);

    transport = std::make_unique<RdmaTransport>(cmId);
    transport->postReceive();
    rpc = std::make_unique<RpcService>(*transport);

    if (rdma_connect(cmId, nullptr))
        throw std::runtime_error("rdma_connect failed");
    wait_event(ec, cmId, RDMA_CM_EVENT_ESTABLISHED);
    printf("Connected to server\n");
}

std::string RpcClientWrapper::echo(const std::string& msg) {
    auto resp = rpc->call(RPC_ECHO, msg.data(), msg.size());
    return std::string(resp.begin(), resp.end());
}

int RpcClientWrapper::add(int a, int b) {
    int nums[2] = {a, b};
    auto resp = rpc->call(RPC_ADD, nums, sizeof(nums));
    if (resp.size() != sizeof(int))
        throw std::runtime_error("Invalid add response size");

    int sum;
    std::memcpy(&sum, resp.data(), sizeof(sum));
    return sum;
}

RpcClientWrapper::~RpcClientWrapper() {
    std::fprintf(stderr, "RpcClientWrapper destructor called\n");
    rdma_disconnect(cmId);
    rdma_destroy_id(cmId);
    rdma_destroy_event_channel(ec);
}

