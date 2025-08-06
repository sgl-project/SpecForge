import torch
import torch.distributed as dist
from transformers import AutoTokenizer, GptOssForCausalLM


@torch.no_grad()
def main():
    dist.init_process_group(backend="nccl")
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    # model = GptOssForCausalLM.from_pretrained("openai/gpt-oss-20b", tp_plan="auto")

    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {num_params / 1e9:.2f}B")

    prompt = "Can I help"
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I'm sorry, I can't help with that."},
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False)
    print(inputs)

    # inputs = tokenizer(inputs, return_tensors="pt").input_ids.to(model.device)
    # outputs = model(inputs)
    # print(outputs)


if __name__ == "__main__":
    main()
