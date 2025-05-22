#include "rpc_handlers.h"
#include <cstring>
#include <cstdio>

bool SendRecvNCCL::initialize(uint32_t rank) {
    if (initialized_) return true;
        
    rank_ = rank;
    
    ncclUniqueId unique_id;
    memset(&unique_id, 0, sizeof(unique_id));
    strcpy((char*)&unique_id, "sendrecv_exchange");
    
    ncclResult_t result = ncclCommInitRank(&comm_, 2, unique_id, rank);
    if (result != ncclSuccess) {
        printf("SendRecv NCCL init failed for rank %d: %s\n", rank, ncclGetErrorString(result));
        return false;
    }
    
    cudaStreamCreate(&stream_);
    initialized_ = true;
    printf("SendRecv NCCL initialized for rank %d\n", rank);
    return true;
}

bool SendRecvNCCL::exchangeData(void* send_buffer, size_t send_count, 
    void* recv_buffer, size_t recv_count, uint32_t peer_rank) {
    if (!initialized_) return false;

    printf("Executing SendRecv exchange with rank %d:\n", peer_rank);
    printf("  Sending: %zu fp16 elements from %p\n", send_count, send_buffer);
    printf("  Receiving: %zu fp16 elements to %p\n", recv_count, recv_buffer);

    try {
        ncclGroupStart();
        ncclSend(send_buffer, send_count, ncclFloat16, peer_rank, comm_, stream_);
        ncclRecv(recv_buffer, recv_count, ncclFloat16, peer_rank, comm_, stream_);
        ncclGroupEnd();

        cudaStreamSynchronize(stream_);
        printf("SendRecv exchange completed successfully\n");
        return true;

    } catch (...) {
        printf("SendRecv exchange failed\n");
        return false;
    }
}

std::vector<char> handleEcho(const char* in, uint16_t len) {
    return std::vector<char>(in, in + len);
}

std::vector<char> handleAdd(const char* in, uint16_t len) {
    if (len < 2 * sizeof(int)) return {};
    int a, b;
    std::memcpy(&a, in,             sizeof(a));
    std::memcpy(&b, in + sizeof(a), sizeof(b));
    int sum = a + b;

    std::vector<char> out(sizeof(sum));
    std::memcpy(out.data(), &sum, sizeof(sum));
    return out;
}

std::vector<char> handleMultThree(const char* in, uint16_t len) {
    if (len < 3 * sizeof(int)) return {};
    int a, b, c;
    std::memcpy(&a, in,             sizeof(a));
    std::memcpy(&b, in + sizeof(a), sizeof(b));
    std::memcpy(&c, in + sizeof(a) + sizeof(b), sizeof(c));
    int mul = a * b * c;

    std::vector<char> out(sizeof(mul));
    std::memcpy(out.data(), &mul, sizeof(mul));
    return out;
}

std::vector<char> handleSendRecv(const char* in, uint16_t len) {

    SendRecvResponse response = {};
    SendRecvRequest request;

    std::memcpy(&request, in, sizeof(request));
    response.request_id = request.request_id;

    printf("Request details:\n");
    printf("Client sending: %lu fp16 elements\n", request.send_size);
    printf("Client receiving: %lu fp16 elements\n", request.recv_size);

    if (!SendRecvNCCL::getInstance().initialize(0)) {
        response.status = 1;
        strcpy(response.message, "Server NCCL init failed");
        std::vector<char> out(sizeof(response));
        std::memcpy(out.data(), &response, sizeof(response));
        return out;
    }

    void* server_recv_buffer = nullptr;
    void* server_send_buffer = nullptr;

    cudaError_t cuda_result = cudaMalloc(&server_recv_buffer, request.send_size * sizeof(__half));
    if (cuda_result != cudaSuccess) {
        response.status = 1;
        strcpy(response.message, "Server send alloc failed");
        cudaFree(server_recv_buffer);
        std::vector<char> out(sizeof(response));
        std::memcpy(out.data(), &response, sizeof(response));
        return out;
    }

    cudaMemset(server_send_buffer, 0xCD, request.recv_size * sizeof(__half));
    printf("Recv buffer: %p (%lu elements)\n", server_recv_buffer, request.send_size);
    printf("Send buffer: %p (%lu elements)\n", server_send_buffer, request.recv_size);

    response.status = 0;

    std::thread([=]() {
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        bool success = SendRecvNCCL::getInstance().exchangeData(
            server_send_buffer, request.recv_size,  
            server_recv_buffer, request.send_size, 
            1
        );
        
        if (success) {
            printf("Server-side exchange completed successfully!\n");
            printf("Received %lu elements from client\n", request.send_size);
            printf("Sent %lu elements to client\n", request.recv_size);
        } else {
            printf("Server-side exchange failed\n");
        }
        
        cudaFree(server_recv_buffer);
        cudaFree(server_send_buffer);
        
    }).detach();

    std::vector<char> out(sizeof(response));
    std::memcpy(out.data(), &response, sizeof(response));
    return out;
}

std::vector<char> dispatch_rpc(uint16_t opcode, const char* payload, uint16_t len) {
    switch (opcode) {
        case RPC_ECHO:
            return handleEcho(payload, len);
        case RPC_ADD:
            return handleAdd(payload, len);
        case RPC_MUL:
            return handleMultThree(payload, len);
        case RPC_NCCL_TRANS:
            return handleSendRecv(payload, len);
        default:
            return {};  // or throw if preferred
    }
}
