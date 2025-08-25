import os
import torch
import torch.distributed as dist
from transformers.models.phi3.configuration_phi3 import Phi3Config

# Set up minimal distributed env for single process
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29504"

# Test without distributed setup first
config = Phi3Config(
    vocab_size=32064,
    hidden_size=3072,
    intermediate_size=8192,
    num_hidden_layers=2,
    max_position_embeddings=4096,
    num_attention_heads=32,
    num_key_value_heads=32,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
    resid_pdrop=0.0,
)

print("Testing standard HF model...")
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM
model = Phi3ForCausalLM(config)

input_ids = torch.randint(0, 1000, (1, 256))
attention_mask = torch.ones_like(input_ids)

try:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    print(f"HF model works fine, logits shape: {logits.shape}")
except Exception as e:
    print(f"HF model failed: {e}")
    import traceback
    traceback.print_exc()

print("\nInitializing distributed for single process...")
from specforge.distributed import init_distributed
init_distributed(tp_size=1)

print("Testing our distributed model (with distribution)...")
try:
    from specforge.modeling.target.phi3 import Phi3ForCausalLM as DistPhi3ForCausalLM
    dist_model = DistPhi3ForCausalLM(config).cuda()
    
    print("Model created successfully")
    
    # Try forward pass with GPU tensors
    input_ids_gpu = input_ids.cuda()
    attention_mask_gpu = attention_mask.cuda()
    logits = dist_model(input_ids=input_ids_gpu, attention_mask=attention_mask_gpu).logits
    print(f"Distributed model works fine, logits shape: {logits.shape}")
    
except Exception as e:
    print(f"Distributed model failed: {e}")
    import traceback
    traceback.print_exc()
