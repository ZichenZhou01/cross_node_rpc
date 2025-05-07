// rpc_rdma.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <stdexcept>
#include <signal.h>
#include <netdb.h>
#include <unistd.h>            // for getopt()
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include "rpc_helper.h"

std::atomic<bool> stop_flag;
std::atomic<uint32_t> next_req_id;

void signal_handler(int) {
    stop_flag = true;
}

// --- RDMA Transport Layer --------------------------------

RdmaTransport::RdmaTransport(rdma_cm_id* id, size_t buf_size)
    : cmId(id), buf(buf_size) {
    init();
}

RdmaTransport::~RdmaTransport() {
    if (mr)     ibv_dereg_mr(mr);
    if (qp)     rdma_destroy_qp(cmId);
    if (sendCq) ibv_destroy_cq(sendCq);
    if (recvCq) ibv_destroy_cq(recvCq);
    if (pd)     ibv_dealloc_pd(pd);
    if (cmId) {
        rdma_disconnect(cmId);
        rdma_destroy_id(cmId);
    }
    if (ec)     rdma_destroy_event_channel(ec);
}

ibv_cq* RdmaTransport::getSendCq() const { return sendCq; }
ibv_cq* RdmaTransport::getRecvCq() const { return recvCq; }
size_t RdmaTransport::capacity() const { return buf.size(); }
char* RdmaTransport::data() { return buf.data(); }

void RdmaTransport::postReceive() {
    ibv_sge sge{};
    sge.addr   = reinterpret_cast<uintptr_t>(buf.data());
    sge.length = buf.size();
    sge.lkey   = mr->lkey;

    ibv_recv_wr wr{};
    wr.wr_id   = reinterpret_cast<uintptr_t>(this);
    wr.next    = nullptr;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    ibv_recv_wr* bad;
    int ret = ibv_post_recv(qp, &wr, &bad);
    if (ret)
        throw std::runtime_error("ibv_post_recv failed: " + std::to_string(ret));
}

void RdmaTransport::postSend(const void* payload, size_t len) {
    ibv_sge sge{};
    sge.addr   = reinterpret_cast<uintptr_t>(payload);
    sge.length = len;
    sge.lkey   = mr->lkey; // ⚠ Assumes payload is in registered memory

    ibv_send_wr wr{};
    wr.wr_id      = reinterpret_cast<uintptr_t>(this);
    wr.next       = nullptr;
    wr.sg_list    = &sge;
    wr.num_sge    = 1;
    wr.opcode     = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;

    ibv_send_wr* bad;
    int ret = ibv_post_send(qp, &wr, &bad);
    if (ret)
        throw std::runtime_error("ibv_post_send failed: " + std::to_string(ret));
}

void RdmaTransport::pollCompletion(ibv_cq* cq) {
    ibv_wc wc;
    while (true) {
        int ne = ibv_poll_cq(cq, 1, &wc);
        if (ne < 0)
            throw std::runtime_error("ibv_poll_cq error: " + std::to_string(ne));
        if (ne == 0)
            continue;
        if (wc.status != IBV_WC_SUCCESS)
            throw std::runtime_error("CQ completion error: " + std::to_string(wc.status));
        return;
    }
}

void RdmaTransport::init() {
    ec = rdma_create_event_channel();
    if (!ec)
        throw std::runtime_error("rdma_create_event_channel failed");

    pd = ibv_alloc_pd(cmId->verbs);
    if (!pd)
        throw std::runtime_error("ibv_alloc_pd failed");

    const int cqe = 1;
    sendCq = ibv_create_cq(cmId->verbs, cqe, nullptr, nullptr, 0);
    if (!sendCq)
        throw std::runtime_error("ibv_create_cq(send) failed");

    recvCq = ibv_create_cq(cmId->verbs, cqe, nullptr, nullptr, 0);
    if (!recvCq)
        throw std::runtime_error("ibv_create_cq(recv) failed");

    ibv_qp_init_attr qpAttr{};
    qpAttr.send_cq = sendCq;
    qpAttr.recv_cq = recvCq;
    qpAttr.cap.max_send_wr  = cqe;
    qpAttr.cap.max_recv_wr  = cqe;
    qpAttr.cap.max_send_sge = 1;
    qpAttr.cap.max_recv_sge = 1;
    qpAttr.qp_type          = IBV_QPT_RC;

    if (rdma_create_qp(cmId, pd, &qpAttr))
        throw std::runtime_error("rdma_create_qp failed");

    qp = cmId->qp;

    mr = ibv_reg_mr(pd, buf.data(), buf.size(),
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr)
        throw std::runtime_error("ibv_reg_mr failed");
}

RpcService::RpcService(RdmaTransport& t)
    : transport(t), replyBuf(t.capacity()) {}

std::vector<char> RpcService::call(uint16_t opcode, const void* payload, uint16_t len) {
    if (len + sizeof(RpcHeader) > transport.capacity())
        throw std::runtime_error("RPC payload too large");

    std::vector<char> req(sizeof(RpcHeader) + len);
    RpcHeader hdr{};
    hdr.request_id  = next_req_id++;
    hdr.opcode      = opcode;
    hdr.payload_len = len;

    std::memcpy(req.data(), &hdr, sizeof(hdr));
    std::memcpy(req.data() + sizeof(hdr), payload, len);

    transport.postReceive();
    transport.postSend(req.data(), req.size());
    transport.pollCompletion(transport.getSendCq());
    transport.pollCompletion(transport.getRecvCq());

    char* buf = transport.data();
    RpcHeader* respHdr = reinterpret_cast<RpcHeader*>(buf);

    if (respHdr->payload_len > transport.capacity())
        throw std::runtime_error("Response payload too large");

    uint16_t outLen = respHdr->payload_len;
    replyBuf.resize(outLen);
    std::memcpy(replyBuf.data(), buf + sizeof(RpcHeader), outLen);
    return replyBuf;
}

void RpcService::serve_loop() {
    while (!stop_flag.load()) {
        transport.postReceive();
        transport.pollCompletion(transport.getRecvCq());

        char* buf = transport.data();
        auto* reqHdr = reinterpret_cast<RpcHeader*>(buf);
        uint16_t inLen = reqHdr->payload_len;

        if (inLen + sizeof(RpcHeader) > transport.capacity())
            continue;

        char* inPayload = buf + sizeof(RpcHeader);
        std::vector<char> out;

        switch (reqHdr->opcode) {
            case RPC_ECHO:
                out = handleEcho(inPayload, inLen);
                break;
            case RPC_ADD:
                out = handleAdd(inPayload, inLen);
                break;
            default:
                continue;
        }

        if (out.size() > std::numeric_limits<uint16_t>::max())
            throw std::runtime_error("Response too large");

        RpcHeader respHdr{};
        respHdr.request_id  = reqHdr->request_id;
        respHdr.opcode      = reqHdr->opcode;
        respHdr.payload_len = static_cast<uint16_t>(out.size());

        std::memcpy(buf, &respHdr, sizeof(respHdr));
        std::memcpy(buf + sizeof(respHdr), out.data(), out.size());

        transport.postSend(buf, sizeof(respHdr) + out.size());
        transport.pollCompletion(transport.getSendCq());
    }
}

std::vector<char> RpcService::handleEcho(const char* in, uint16_t len) {
    return std::vector<char>(in, in + len);
}

std::vector<char> RpcService::handleAdd(const char* in, uint16_t len) {
    if (len < 2 * sizeof(int)) return {};
    int a, b;
    std::memcpy(&a, in,             sizeof(a));
    std::memcpy(&b, in + sizeof(a), sizeof(b));
    int sum = a + b;

    std::vector<char> out(sizeof(sum));
    std::memcpy(out.data(), &sum, sizeof(sum));
    return out;
}


// --- Event‐helper --------------------------------
void wait_event(rdma_event_channel* ec, rdma_cm_id* id, rdma_cm_event_type expect) {
    rdma_cm_event* ev;
    while (!stop_flag && rdma_get_cm_event(ec, &ev) == 0) {
        if (ev->id == id && ev->event == expect) {
            rdma_ack_cm_event(ev);
            return;
        }
        rdma_ack_cm_event(ev);
    }
}

// --- run_server & run_client --------------------------------
int run_server(const char* port) {
    signal(SIGINT, signal_handler);

    rdma_event_channel* ec = rdma_create_event_channel();
    if (!ec) throw std::runtime_error("rdma_create_event_channel");
    rdma_cm_id* listener;
    if (rdma_create_id(ec, &listener, nullptr, RDMA_PS_TCP))
        throw std::runtime_error("rdma_create_id");

    struct addrinfo hints = {};
    hints.ai_flags    = AI_PASSIVE;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res;
    if (getaddrinfo(nullptr, port, &hints, &res))
        throw std::runtime_error("getaddrinfo");
    if (rdma_bind_addr(listener, res->ai_addr))
        throw std::runtime_error("rdma_bind_addr");
    freeaddrinfo(res);
    if (rdma_listen(listener, 0))
        throw std::runtime_error("rdma_listen");

    while (!stop_flag) {
        rdma_cm_event* ev;
        if (rdma_get_cm_event(ec, &ev)) break;
        if (ev->event == RDMA_CM_EVENT_CONNECT_REQUEST) {
            rdma_cm_id* clientId = ev->id;
            rdma_ack_cm_event(ev);

            auto* ctx = new RdmaTransport(clientId);
            RpcService rpc(*ctx);
            ctx->postReceive();
            
            if (rdma_accept(clientId, nullptr))
                throw std::runtime_error("rdma_accept failed");
            wait_event(ec, clientId, RDMA_CM_EVENT_ESTABLISHED);

            std::thread([ctx,&rpc](){
                rpc.serve_loop();
                delete ctx;
            }).detach();
        } else {
            rdma_ack_cm_event(ev);
        }
    }

    rdma_destroy_id(listener);
    rdma_destroy_event_channel(ec);
    return 0;
}

int run_client(const char* host, const char* port) {
    signal(SIGINT, signal_handler);

    rdma_event_channel* ec = rdma_create_event_channel();
    if (!ec) throw std::runtime_error("rdma_create_event_channel");

    rdma_cm_id* cmId;
    if (rdma_create_id(ec, &cmId, nullptr, RDMA_PS_TCP))
        throw std::runtime_error("rdma_create_id");

    struct addrinfo hints = {};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res;
    if (getaddrinfo(host, port, &hints, &res))
        throw std::runtime_error("getaddrinfo");
    if (rdma_resolve_addr(cmId, nullptr, res->ai_addr, 2000))
        throw std::runtime_error("rdma_resolve_addr");
    freeaddrinfo(res);

    wait_event(ec, cmId, RDMA_CM_EVENT_ADDR_RESOLVED);
    if (rdma_resolve_route(cmId, 2000))
        throw std::runtime_error("rdma_resolve_route");
    wait_event(ec, cmId, RDMA_CM_EVENT_ROUTE_RESOLVED);

    RdmaTransport transport(cmId);
    transport.postReceive();   
    RpcService   rpc(transport);

    if (rdma_connect(cmId, nullptr))
        throw std::runtime_error("rdma_connect");
    wait_event(ec, cmId, RDMA_CM_EVENT_ESTABLISHED);
    for (size_t i = 0; i < 20; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        const char* msg = "Hello RDMA RPC";
        auto echoResp = rpc.call(RPC_ECHO, msg, std::strlen(msg));
    
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::printf("Echo: %.*s\n", (int)echoResp.size(), echoResp.data());
        std::printf("RPC call duration: %ld microseconds\n", duration);
    }


    int nums[2] = {7,5};
    auto addResp = rpc.call(RPC_ADD, nums, sizeof(nums));
    if (addResp.size() == sizeof(int)) {
        int sum;
        std::memcpy(&sum, addResp.data(), sizeof(sum));
        std::printf("Add: 7 + 5 = %d\n", sum);
    }

    return 0;
}

// int main(int argc, char** argv) {
//     bool isServer = false;
//     const char* host = nullptr;
//     const char* port = "7471";
//     int c;
//     while ((c = getopt(argc, argv, "sd:p:")) != -1) {
//         switch (c) {
//         case 's': isServer = true;   break;
//         case 'd': host     = optarg; break;
//         case 'p': port     = optarg; break;
//         default:
//             std::fprintf(stderr, "Usage: %s [-s] [-d <addr>] [-p <port>]\n", argv[0]);
//             return EXIT_FAILURE;
//         }
//     }

//     try {
//         if (isServer)      return run_server(port);
//         else if (host)     return run_client(host, port);
//         else {
//             std::fprintf(stderr, "Client mode requires -d <server>\n");
//             return EXIT_FAILURE;
//         }
//     } catch (const std::exception& ex) {
//         std::fprintf(stderr, "ERROR: %s\n", ex.what());
//         return EXIT_FAILURE;
//     }
// }
