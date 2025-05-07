# Example library interface python rdma with RPC

A quick demo of using RDMA interface to allow working across nodes in c++ via pytorch extension.

## Installation
```
python3 setup.py develop
```
This will compile the C++ using pytorch extension. This can be used via:
```
import torch
import rpc_rdma
```

rdma/infiband needs to be setup on the servers.

### Running the Cross-Node setup
Check that the RDMA links are available
```
rdma link show
```
```
ipaddr show
```
Find the relevant IP addr on the client server

Machine 0: Run
```
python3 rpc_server.py
```

Machine 1: Run
```
python3 client.py
```
Change the IP address within test_client to use the ip address for the other machine


## Basic Profiling
Quick sanity check on RDMA vs TCP performance
### Using RDMA RPC
I notice around 6-7 microseconds for a basic rpc call:
```
Time in microseconds: 7.62939453125
Time in microseconds: 6.9141387939453125
```

### Using ZeroMQ over TCP
```
Time in microseconds: 226.73606872558594
Time in microseconds: 222.68295288085938
```

## Adding more RPC
### Server Side:
```
csrc/rpc_handler.cpp
csrc/rpc_handler.h
```
To add a new rpc call. I've currently added a basic Add/Expo Call

### Client Side:
In 
```
rpc_wrapper_binding.cpp
```
Add a new function that represnts the new RPC call

#### Exposing them via pybind
update `python_bindings.cpp` with the relevant bindings. Any class here will be exposed via python