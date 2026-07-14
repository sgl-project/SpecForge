import os
import socket
import subprocess
import time

import requests
from sglang.utils import print_highlight


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
    """Execute a shell command and return its process handle."""
    command = command.replace("\\\n", " ").replace("\\", " ")
    env = os.environ.copy()

    if disable_proxy:
        for name in (
            "http_proxy",
            "https_proxy",
            "no_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
        ):
            env.pop(name, None)

    if enable_hf_mirror:
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
    return subprocess.Popen(
        command.split(), text=True, stderr=subprocess.STDOUT, env=env
    )


def wait_for_server(
    base_url: str, timeout: int | None = None, disable_proxy: bool = False
) -> None:
    """Wait until a server's OpenAI-compatible models endpoint is ready."""
    started = time.perf_counter()
    proxy_names = (
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
    )
    saved_proxies = {}
    if disable_proxy:
        saved_proxies = {
            name: os.environ.pop(name) for name in proxy_names if name in os.environ
        }

    try:
        while True:
            try:
                response = requests.get(
                    f"{base_url}/v1/models",
                    headers={"Authorization": "Bearer None"},
                    timeout=5,
                )
                if response.status_code == 200:
                    time.sleep(5)
                    print_highlight(
                        "Server is ready; server and test output may be interleaved."
                    )
                    return
            except requests.exceptions.RequestException:
                pass

            if timeout is not None and time.perf_counter() - started > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
            time.sleep(1)
    finally:
        os.environ.update(saved_proxies)
