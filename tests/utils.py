import os
import socket
import subprocess


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


def execute_shell_command(
    command: str, disable_proxy: bool = False, enable_hf_mirror: bool = False
):
    """
    Execute a shell command and return its process handle.
    """
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    env = os.environ.copy()

    if disable_proxy:
        env.pop("http_proxy", None)
        env.pop("https_proxy", None)
        env.pop("no_proxy", None)
        env.pop("HTTP_PROXY", None)
        env.pop("HTTPS_PROXY", None)
        env.pop("NO_PROXY", None)

    if enable_hf_mirror:
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
    return subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT, env=env)
