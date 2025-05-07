"""
RDMA-based RPC using pybind11
"""

from __future__ import annotations

__all__ = ["Client", "start_server"]

class Client:
    def __init__(self, host: str, port: str = "7471") -> None: ...
    def add(self, arg0: int, arg1: int) -> int:
        """
        Send an add RPC and receive the sum
        """

    def echo(self, arg0: str) -> str:
        """
        Send an echo RPC and receive the response
        """

def start_server(port: str = "7471") -> None:
    """
    Start the RDMA RPC server in a background thread
    """
