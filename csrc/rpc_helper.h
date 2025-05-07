#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

// Signal handler to set stop flag
void signal_handler(int signum);

// Helper to wait for RDMA CM events
void wait_event(rdma_event_channel* ec, rdma_cm_id* id, rdma_cm_event_type expect);

#pragma pack(push,1)
// RPC message header
struct RpcHeader {
    uint32_t request_id;
    uint16_t opcode;
    uint16_t payload_len;
};
#pragma pack(pop)

// RPC operations
enum {
    RPC_ECHO = 1,
    RPC_ADD  = 2
};

// Low-level RDMA transport
class RdmaTransport {
public:
    RdmaTransport(rdma_cm_id* id, size_t buf_size = 1024);
    ~RdmaTransport();

    ibv_cq* getSendCq() const;
    ibv_cq* getRecvCq() const;
    size_t capacity() const;
    char* data();

    void postReceive();
    void postSend(const void* payload, size_t len);
    void pollCompletion(ibv_cq* cq);

private:
    void init();

    rdma_cm_id*          cmId;
    rdma_event_channel*  ec;
    ibv_pd*              pd;
    ibv_cq*              sendCq;
    ibv_cq*              recvCq;
    ibv_qp*              qp;
    ibv_mr*              mr;
    std::vector<char>    buf;
};

// RPC service layer atop RdmaTransport
class RpcService {
public:
    explicit RpcService(RdmaTransport& t);

    // Perform blocking RPC call
    std::vector<char> call(uint16_t opcode, const void* payload, uint16_t len);

    // Serve incoming RPC requests until stop_flag is true
    void serve_loop();

private:
    std::vector<char> handleEcho(const char* in, uint16_t len);
    std::vector<char> handleAdd(const char* in, uint16_t len);

    RdmaTransport& transport;
    std::vector<char> replyBuf;
};

// Entry points
int run_server(const char* port);
int run_client(const char* host, const char* port);
