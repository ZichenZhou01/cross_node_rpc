#pragma once
#include <vector>
#include <cstdint>
#include <thread>
#include <chrono>
#include <netdb.h>
#include <signal.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nccl.h>

enum RpcOpcode : uint16_t {
    RPC_ECHO = 1,
    RPC_ADD  = 2,
    RPC_MUL  = 3,
    RPC_NCCL_TRANS = 4,
    // Add more opcodes as needed
};

struct SendRecvRequest {
    uint32_t request_id;
    uint64_t send_size;     
    uint64_t recv_size;      
    uint64_t send_gpu_addr;   
    uint64_t recv_gpu_addr;    
};

struct SendRecvResponse {
    uint32_t request_id;
    uint32_t status;     
    char message[64];
    uint64_t allocated_recv_addr;  
};

class SendRecvNCCL {

    public:
        SendRecvNCCL() : initialized_(false), rank_(0) {}
        bool initialize();
        bool exchangeData(void* send_buffer, size_t send_count, 
            void* recv_buffer, size_t recv_count, uint32_t peer_rank);
        
        static SendRecvNCCL& getInstance() {
            static SendRecvNCCL instance;
            return instance;
        }

    private:
        ncclComm_t comm_;
        cudaStream_t stream_;
        bool initialized_;
        uint32_t rank_;
};

std::vector<char> handleEcho(const char* in, uint16_t len);
std::vector<char> handleAdd(const char* in, uint16_t len);
std::vector<char> handleMultThree(const char* in, uint16_t len);
std::vector<char> handleSendRecv(const char* in, uint16_t len);
std::vector<char> dispatch_rpc(uint16_t opcode, const char* payload, uint16_t len);