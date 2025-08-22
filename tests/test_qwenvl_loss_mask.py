from transformers import AutoProcessor

from specforge.data.preprocessing import preprocess_vlm_conversations
from specforge.data.template import TEMPLATE_REGISTRY

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def test_preprocess_vlm_conversations():
    conversations = [
        {"role": "user", "content": "what is in the image?"},
        {"role": "assistant", "content": "This is an image of a cat."},
    ]

    examples = {
        "id": ["example1"],
        "image": [
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        ],
        "conversations": [conversations],
    }
    # examples = [examples]  # Wrap in a list to match expected input format
    chat_template = TEMPLATE_REGISTRY.get("qwen2-vl")
    processed = preprocess_vlm_conversations(processor, examples, chat_template, 4096)

    input_ids = processed["input_ids"][0]
    print(f"Loss mask sum: {processed['loss_mask'][0].sum()}")
    loss_mask = processed["loss_mask"][0].squeeze(0).tolist()
    input_ids = input_ids.squeeze(0)
    current_mask = input_ids[0]
    current_ids = []

    for i in range(len(input_ids)):
        # for i in range(input_ids.shape[-1]):
        if current_mask == loss_mask[i]:
            current_ids.append(input_ids[i])
        else:
            decoded_text = processor.tokenizer.decode(
                current_ids, skip_special_tokens=False
            )
            if current_mask == 0:
                print(f"{RED}{decoded_text}{RESET}", end="")
            else:
                print(f"{GREEN}{decoded_text}{RESET}", end="")
            current_ids = [input_ids[i]]
            current_mask = loss_mask[i]

    print(
        f"{GREEN}{processor.tokenizer.decode(current_ids, skip_special_tokens=False)}{RESET}"
    )

    print()
    print(f"input_ids shape: {processed['input_ids'][0].shape}")
    print(f"loss_mask shape: {processed['loss_mask'][0].shape}")
    # print(f"hidden_state shape: {processed['hidden_state'].shape}")


if __name__ == "__main__":
    test_preprocess_vlm_conversations()
