#pragma once
#include <vector>
#include <cstdint>

enum RpcOpcode : uint16_t {
    RPC_ECHO = 1,
    RPC_ADD  = 2,
    // Add more opcodes as needed
};

std::vector<char> handleEcho(const char* in, uint16_t len);
std::vector<char> handleAdd(const char* in, uint16_t len);
std::vector<char> dispatch_rpc(uint16_t opcode, const char* payload, uint16_t len);