#include "rpc_handlers.h"
#include <cstring>

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

std::vector<char> dispatch_rpc(uint16_t opcode, const char* payload, uint16_t len) {
    switch (opcode) {
        case RPC_ECHO:
            return handleEcho(payload, len);
        case RPC_ADD:
            return handleAdd(payload, len);
        case RPC_MUL:
            return handleMultThree(payload, len);
        default:
            return {};  // or throw if preferred
    }
}
