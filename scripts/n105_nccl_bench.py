import os, time, torch, torch.distributed as dist
dist.init_process_group('nccl')
r = dist.get_rank(); torch.cuda.set_device(r)
N = 5 * (1 << 30) // 2  # 5 GiB bf16 output per rank-group op
out = torch.empty(N, dtype=torch.bfloat16, device='cuda')
shard = torch.randn(N // 8, dtype=torch.bfloat16, device='cuda')
for _ in range(3): dist.all_gather_into_tensor(out, shard)
torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
for _ in range(10): dist.all_gather_into_tensor(out, shard)
torch.cuda.synchronize(); dt = time.time() - t0
if r == 0: print(f"all_gather 5GiB x10: {dt:.2f}s -> algbw {10*5/dt:.0f} GiB/s")
g = torch.randn(N, dtype=torch.bfloat16, device='cuda'); o = torch.empty(N // 8, dtype=torch.bfloat16, device='cuda')
for _ in range(3): dist.reduce_scatter_tensor(o, g)
torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
for _ in range(10): dist.reduce_scatter_tensor(o, g)
torch.cuda.synchronize(); dt = time.time() - t0
if r == 0: print(f"reduce_scatter 5GiB x10: {dt:.2f}s -> algbw {10*5/dt:.0f} GiB/s")
dist.destroy_process_group()
