import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import socket
import statistics
import subprocess
import sys
import time


DATASETS = ("gsm8k", "math500", "humaneval", "mbpp", "mt-bench")


def speculative_server_args(algorithm, width, verify_width=None):
    if algorithm == "DSPARK":
        return ["--speculative-algorithm", "DSPARK", "--speculative-dspark-block-size", str(width if verify_width is not None else width - 1)]
    if algorithm == "DFLASH":
        return ["--speculative-algorithm", "DFLASH", "--speculative-dflash-block-size", str(width)]
    raise ValueError(algorithm)


def validate_verify_budget(records, width):
    for record in records:
        if record["verify_count"] <= 0 or not 0 < record["completion_tokens"] <= 1 + width * record["verify_count"]:
            raise ValueError("Completion count exceeds prefill token plus verification budget")


def aggregate(records):
    if not records:
        raise ValueError("Cannot aggregate empty records")
    for record in records:
        if record["verify_count"] <= 0 or record["completion_tokens"] <= 0:
            raise ValueError("Missing positive speculative counters")
        ratio = record["completion_tokens"] / record["verify_count"]
        if not math.isclose(ratio, record["accept_length"], rel_tol=1e-6):
            raise ValueError("Server acceptance length disagrees with counters")
    tokens = sum(record["completion_tokens"] for record in records)
    verifies = sum(record["verify_count"] for record in records)
    return {
        "requests": len(records),
        "output_tokens": tokens,
        "verify_count": verifies,
        "macro_accept_length": statistics.fmean(record["accept_length"] for record in records),
        "micro_accept_length": tokens / verifies,
        "length_limited_requests": sum(record["finish_reason"].get("type") == "length" for record in records),
    }


def prepare(root):
    data_root = root / "data"
    data_root.mkdir(exist_ok=True)
    manifest = {}
    for name in DATASETS:
        path = data_root / (name + ".json")
        if not path.exists():
            from specforge.benchmarks.sglang import _load_prompts

            prompts = _load_prompts(name, None)
            path.write_text(json.dumps(prompts, ensure_ascii=False))
        prompts = json.loads(path.read_text())
        manifest[name] = {
            "conversations": len(prompts),
            "requests": sum(len(turns) for turns in prompts),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print("DATA", name, manifest[name], flush=True)
    (root / "data_manifest.json").write_text(json.dumps(manifest, indent=2))


def generate(endpoint, tokenizer, turns, name, sample_id, max_tokens):
    import requests

    messages = []
    records = []
    for turn_id, question in enumerate(turns):
        messages.append({"role": "user", "content": question})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        started = time.monotonic()
        response = requests.post(endpoint + "/generate", json={
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_new_tokens": max_tokens},
        }, timeout=600)
        response.raise_for_status()
        output = response.json()
        metadata = output["meta_info"]
        record = {
            "dataset": name, "sample_id": sample_id, "turn": turn_id + 1,
            "endpoint": endpoint, "completion_tokens": metadata["completion_tokens"],
            "verify_count": metadata["spec_verify_ct"],
            "accept_length": metadata["spec_accept_length"],
            "finish_reason": metadata.get("finish_reason", {}),
            "latency_seconds": time.monotonic() - started,
            "output": output["text"], "meta_info": metadata,
        }
        aggregate([record])
        records.append(record)
        messages.append({"role": "assistant", "content": output["text"]})
    return records


def evaluate(root, endpoints, max_tokens):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("/mnt/checkpoint/Qwen3-8B", local_files_only=True)
    for endpoint in endpoints:
        generate(endpoint, tokenizer, ["Say hello briefly."], "warmup", 0, 32)
    results = {}
    for name in DATASETS:
        prompts = json.loads((root / "data" / (name + ".json")).read_text())
        output_path = root / (name + ".jsonl")
        records = []
        if output_path.exists():
            records = [json.loads(line) for line in output_path.read_text().splitlines() if line]
        completed = {record["sample_id"] for record in records}
        tasks = queue.Queue()
        for sample_id, turns in enumerate(prompts):
            if sample_id not in completed:
                tasks.put((sample_id, turns))

        def worker(endpoint):
            while True:
                try:
                    sample_id, turns = tasks.get_nowait()
                except queue.Empty:
                    return
                yield generate(endpoint, tokenizer, turns, name, sample_id, max_tokens)

        with output_path.open("a", buffering=1) as output_file:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
                streams = {executor.submit(next, stream, None): stream for stream in map(worker, endpoints)}
                while streams:
                    done, _ = concurrent.futures.wait(streams, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        stream = streams.pop(future)
                        batch = future.result()
                        if batch is None:
                            continue
                        for record in batch:
                            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output_file.flush()
                        records.extend(batch)
                        if len(records) % 50 == 0:
                            print("PROGRESS", name, aggregate(records), flush=True)
                        streams[executor.submit(next, stream, None)] = stream
        assert len(records) == sum(len(turns) for turns in prompts)
        results[name] = aggregate(records)
        if name == "mt-bench":
            results[name]["by_turn"] = {
                str(turn): aggregate([record for record in records if record["turn"] == turn])
                for turn in (1, 2)
            }
        (root / "summary.json").write_text(json.dumps(results, indent=2))
        print("RESULT", name, json.dumps(results[name]), flush=True)


def run(args):
    import requests

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prepare(root)
    if args.prepare_only:
        return
    assert (root / "export.done").exists(), "Draft export not ready"
    processes = []
    handles = []
    endpoints = []
    (root / "settings.json").write_text(json.dumps(vars(args), indent=2))
    try:
        for index, gpu in enumerate(args.gpu_uuids):
            used = int(subprocess.check_output([
                "nvidia-smi", "--id=" + gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"
            ], text=True).strip())
            if used > 100:
                raise RuntimeError(f"GPU {gpu} is no longer idle: {used} MiB")
            port = args.base_port + index
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4",
                               FLASHINFER_DISABLE_VERSION_CHECK="1", TOKENIZERS_PARALLELISM="false")
            environment.pop("SPECFORGE_DFLASH_VERIFY_BLOCK_SIZE", None)
            environment.pop("SPECFORGE_DSPARK_VERIFY_WIDTH", None)
            if args.verify_block_size is not None:
                variable = "SPECFORGE_DSPARK_VERIFY_WIDTH" if args.algorithm == "DSPARK" else "SPECFORGE_DFLASH_VERIFY_BLOCK_SIZE"
                environment[variable] = str(args.verify_block_size)
            command = [sys.executable, "-m", "sglang.launch_server",
                       "--model-path", "/mnt/checkpoint/Qwen3-8B",
                       *speculative_server_args(args.algorithm, args.block_size, args.verify_block_size),
                       "--speculative-draft-model-path", str(root / "draft_hf"),
                       "--tp-size", "1", "--dtype", "bfloat16", "--attention-backend", "flashinfer",
                       "--context-length", "16384", "--max-running-requests", "1",
                       "--max-total-tokens", "16384", "--mem-fraction-static", "0.7",
                       "--chunked-prefill-size", "-1", "--disable-radix-cache", "--disable-cuda-graph",
                       "--host", "127.0.0.1", "--port", str(port)]
            if args.domino_candidate_pool_size is not None:
                command.extend(["--speculative-domino-candidate-pool-size",
                                str(args.domino_candidate_pool_size)])
            handle = (root / f"server-{index}.log").open("a")
            handles.append(handle)
            process = subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            endpoints.append(f"http://127.0.0.1:{port}")
        (root / "server_pids.json").write_text(json.dumps([process.pid for process in processes]))
        deadline = time.monotonic() + 1200
        for process, endpoint in zip(processes, endpoints):
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server {process.pid} exited with {process.returncode}")
                try:
                    if requests.get(endpoint + "/health", timeout=3).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(3)
            else:
                raise TimeoutError(endpoint)
        print("SERVERS_READY", endpoints, flush=True)
        server_info = []
        for endpoint in endpoints:
            response = requests.get(endpoint + "/get_server_info", timeout=30)
            response.raise_for_status()
            server_info.append(response.json())
        (root / "server_info.json").write_text(json.dumps(server_info, indent=2))
        evaluate(root, endpoints, args.max_new_tokens)
        (root / "evaluation.done").touch()
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--algorithm", choices=("DFLASH", "DSPARK"), default="DFLASH")
    parser.add_argument("--gpu-uuids", nargs="+", default=[])
    parser.add_argument("--base-port", type=int, default=32140)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--block-size", type=int, choices=(8, 16), default=16)
    parser.add_argument("--verify-block-size", type=int, choices=(8, 16), default=None)
    parser.add_argument("--domino-candidate-pool-size", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.algorithm == "DSPARK" and arguments.verify_block_size is not None and arguments.block_size != 16:
        parser.error("DSPARK split verification requires --block-size 16")
    if arguments.verify_block_size is not None and arguments.verify_block_size > arguments.block_size:
        parser.error("--verify-block-size must not exceed --block-size")
    if not arguments.prepare_only and not arguments.gpu_uuids:
        parser.error("--gpu-uuids is required for evaluation")
    run(arguments)
