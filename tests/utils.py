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


def wait_for_server(
    base_url: str, timeout: int = None, disable_proxy: bool = False
) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.perf_counter()

    if disable_proxy:
        http_proxy = os.environ.pop("http_proxy", None)
        https_proxy = os.environ.pop("https_proxy", None)
        no_proxy = os.environ.pop("no_proxy", None)
        http_proxy_capitalized = os.environ.pop("HTTP_PROXY", None)
        https_proxy_capitalized = os.environ.pop("HTTPS_PROXY", None)
        no_proxy_capitalized = os.environ.pop("NO_PROXY", None)

    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                time.sleep(5)
                print_highlight(
                    """\n
                    NOTE: Typically, the server runs in a separate terminal.
                    In this notebook, we run the server and notebook code together, so their outputs are combined.
                    To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.
                    To reduce the log length, we set the log level to warning for the server, the default log level is info.
                    We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.
                    """
                )
                break

            if timeout and time.perf_counter() - start_time > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            time.sleep(1)

    if disable_proxy:
        if http_proxy:
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["https_proxy"] = https_proxy
        if no_proxy:
            os.environ["no_proxy"] = no_proxy
        if http_proxy_capitalized:
            os.environ["HTTP_PROXY"] = http_proxy_capitalized
        if https_proxy_capitalized:
            os.environ["HTTPS_PROXY"] = https_proxy_capitalized
        if no_proxy_capitalized:
            os.environ["NO_PROXY"] = no_proxy_capitalized
