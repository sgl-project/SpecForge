import socket

import torch
import torch.nn.init


def norm_tensor(shape, device, dtype, std=0.02):
    t = torch.empty(shape, device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.trunc_normal_(t, mean=0.0, std=std)
    return t

def is_port_in_use(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return False
        except OSError:
            return True


def get_available_port():
    # get a random available port
    # and try to find a port that is not in use
    for port in range(10000, 65535):
        if not is_port_in_use(port):
            return port
    raise RuntimeError("No available port found")
