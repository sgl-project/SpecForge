import os
import json
from PIL import Image
from tqdm import tqdm
import torch

from transformers import AutoProcessor
from specforge.hf_model import LlavaForConditionalGeneration
from specforge.eagle3 import LlamaForCausalLMEagle3, EaModel
if __name__ == "__main__":
    # Load model
    model_id = "cache/target_model/llava-1.5-7b-hf"
    target_model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    ).to("cuda:0", dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(model_id)

    draft_model = LlamaForCausalLMEagle3.from_pretrained(
        "cache/draft_model/EAGLE_llava7B"
        ).to("cuda:0", dtype=torch.float16)
    
    model = EaModel(
        target_model=target_model, 
        draft_model=draft_model, 
        tokenizer=processor.tokenizer
        )

    conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "What is unusual about this image? Can you explain this to a 5-year-old kid?"},
          {"type": "image"},
        ],
    },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Load image
    image_file = "demo.jpeg"
    raw_image = Image.open(image_file)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to("cuda:0", dtype=torch.float16)

    # naivegenerate
    print("=" * 50)
    print("Naive Generation Output")
    print("=" * 50)
    output = model.naivegenerate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"]
    )
    print(processor.decode(output, skip_special_tokens=True))
 
    # eagenerate
    print("\n" + "=" * 50)
    print("EA Generation Output")
    print("=" * 50)
    output = model.eagenerate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        log=True
    )
    print(processor.decode(output[0], skip_special_tokens=True))
    print(f"average accepted tokens: {output[-1]}")


