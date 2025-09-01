"""
Simple script to generate responses using local vLLM API from JSONL file.
Usage:
python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 \
    --mem-frac=0.4 \
    --tp 8 --port 30001 --attention-backend fa3 --reasoning-parser gpt-oss
python scripts/generate_data_by_target.py \
    --model-name $MODEL_PATH \
    --raw-data-file $DATASET_PATH/perfectblend.jsonl \
    --output-dir $SHARE_PREFIX/gpt-oss-120b-generated/perfectblend \
    --max-concurrency 512 \
    --num-per-shard 50000 \
    --server-address-port 127.0.0.1:30001 \
    --is-reasoning-model \
    --is-gpt-oss
"""

import argparse
import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from openai import AsyncOpenAI, OpenAI, OpenAIError
from tqdm.asyncio import tqdm

SYSTEM_PROMPT = ""
# SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--num-per-shard", type=int, default=50_000)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=16 * 1024)
    parser.add_argument(
        "--server-address-port", type=str, nargs="+", default=["127.0.0.1:30000"]
    )
    parser.add_argument("--is-reasoning-model", action="store_true")
    parser.add_argument("--is-gpt-oss", action="store_true")
    return parser.parse_args()


@dataclass
class RequestFuncObject:
    conversation_id: str
    input_conversations: List[Dict[str, str]]
    model_name: str
    system_prompt: Optional[str]
    temperature: float = 0.0
    max_tokens: int = 16 * 1024
    output_conversations: Optional[List[Dict[str, str]]] = None
    output_tokens: int = 0
    error: Optional[str] = None
    is_reasoning_model: bool = False
    extra_body: Dict[str, Any] = field(default_factory=dict)


async def build_conversation(
    req_obj: RequestFuncObject, client: AsyncOpenAI, pbar: Optional[tqdm] = None
) -> str:
    messages = []
    if req_obj.system_prompt is not None and len(req_obj.system_prompt) > 0:
        messages.append({"role": "system", "content": req_obj.system_prompt})
    req_obj.output_tokens = 0
    for conversation in req_obj.input_conversations:
        if conversation["role"] == "assistant":
            continue
        if conversation["role"] == "user":
            messages.append({"role": "user", "content": conversation["content"]})
            try:
                response = await client.chat.completions.create(
                    model=req_obj.model_name,
                    messages=messages,
                    max_tokens=req_obj.max_tokens,
                    temperature=req_obj.temperature,
                    stream=False,
                )
                response_text = response.choices[0].message.content
                req_obj.output_tokens += response.usage.completion_tokens
                if req_obj.is_reasoning_model:
                    reasoning_content = response.choices[0].message.reasoning_content
            except Exception as e:
                req_obj.error = str(e)
                break
            msg = {"role": "assistant", "content": response_text}
            if req_obj.is_reasoning_model:
                msg["thinking"] = reasoning_content
            messages.append(msg)

    if pbar is not None:
        pbar.update(1)
    req_obj.output_conversations = messages
    return req_obj


async def limited_build_conversation(
    req_obj: RequestFuncObject,
    client: AsyncOpenAI,
    semaphore: Optional[asyncio.Semaphore] = None,
    pbar: Optional[tqdm] = None,
):
    if semaphore is None:
        return await build_conversation(req_obj=req_obj, client=client, pbar=pbar)
    async with semaphore:
        return await build_conversation(req_obj=req_obj, client=client, pbar=pbar)


def get_random_temperature() -> float:
    choices = [0.0, 0.3, 0.5, 0.7, 1.0]
    weights = [4, 1, 1, 1, 3]
    return random.choices(choices, weights=weights)[0]


def get_random_reasoning_effort() -> str:
    """Get a random reasoning effort level for the model with weighted probabilities."""
    # usage example: https://huggingface.co/openai/gpt-oss-20b/discussions/28
    # Reasoning effort levels with weights: LOW(3), MEDIUM(6), HIGH(1)
    reasoning_efforts = [
        "low",
        "medium",
        "high",
    ]
    weights = [3, 6, 1]
    return random.choices(reasoning_efforts, weights=weights, k=1)[0]


async def main():
    args = parse_args()
    if args.is_gpt_oss:
        args.is_reasoning_model = True
    os.makedirs(args.output_dir, exist_ok=True)

    total_ds = load_dataset("json", data_files=args.raw_data_file)["train"]
    # total_ds = total_ds.select(range(10)) # used to debug
    for start in range(0, len(total_ds), args.num_per_shard):
        end = min(start + args.num_per_shard, len(total_ds))
        output_file = os.path.join(args.output_dir, f"shard_{start}-{end}.jsonl")
        output_file_error = os.path.join(args.output_dir, f"error.jsonl")
        if os.path.exists(output_file):
            print(f"Skipping generate data {output_file} because it already exists")
            continue
        print(f"Generating data {output_file}")
        ds = total_ds.select(range(start, end))
        pbar = None if args.disable_tqdm else tqdm(total=len(ds))
        client_semaphore_list = []
        for server_address_port in args.server_address_port:
            client = AsyncOpenAI(
                base_url=f"http://{server_address_port}/v1", api_key="None"
            )
            semaphore = (
                asyncio.Semaphore(args.max_concurrency)
                if args.max_concurrency
                else None
            )
            client_semaphore_list.append((client, semaphore))

        tasks = []
        for i, row in enumerate(ds):
            client, semaphore = client_semaphore_list[i % len(client_semaphore_list)]
            req_obj = RequestFuncObject(
                conversation_id=str(row["id"]),
                input_conversations=row["conversations"],
                model_name=args.model_name,
                system_prompt=SYSTEM_PROMPT,
                temperature=get_random_temperature(),
                max_tokens=args.max_tokens,
                is_reasoning_model=args.is_reasoning_model,
                extra_body=(
                    {"reasoning_effort": get_random_reasoning_effort()}
                    if args.is_gpt_oss
                    else {}
                ),
            )
            tasks.append(
                asyncio.create_task(
                    limited_build_conversation(
                        req_obj=req_obj, client=client, semaphore=semaphore, pbar=pbar
                    )
                )
            )
        outputs = await asyncio.gather(*tasks)
        with open(output_file, "w") as f, open(output_file_error, "a") as f_error:
            for output_obj in outputs:
                output_dict = {
                    "conversation_id": output_obj.conversation_id,
                    "conversations": output_obj.output_conversations,
                }
                if args.is_gpt_oss:
                    output_dict["reasoning_effort"] = output_obj.extra_body[
                        "reasoning_effort"
                    ]
                if output_obj.error is not None:
                    output_dict["error"] = output_obj.error
                    output_dict["input_conversations"] = output_obj.input_conversations
                    f_error.write(json.dumps(output_dict) + "\n")
                else:
                    f.write(json.dumps(output_dict) + "\n")


if __name__ == "__main__":
    asyncio.run(main())


# from openai import OpenAIError, AsyncOpenAI, OpenAI
# client = OpenAI(base_url=f"http://127.0.0.1:30001/v1", api_key="None")
# messages = []
# def ask(user_text: str, messages, client):
#     messages.append({"role": "user", "content": user_text})
#     resp = client.chat.completions.create(
#         model=MODEL,
#         messages=messages,
#         extra_body={"reasoning_effort": "low"},
#         temperature=0.3,
#         max_tokens=512,
#     )
#     assistant_text = resp.choices[0].message.content
#     reasoning_content = resp.choices[0].message.reasoning_content
#     messages.append({"role": "assistant", "content": assistant_text, "thinking": reasoning_content})
#     return assistant_text

# # --- Multi-turn conversation ---
# print(ask("Give me a two-sentence overview of diffusion models.", messages, client))
# print(ask("Great—now compare them to autoregressive LLMs in 3 bullets.", messages, client))
# print(ask("Suggest one practical use case for each, max 20 words per item.", messages, client))
