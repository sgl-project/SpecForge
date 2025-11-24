import torch
import torch.nn.init


def norm_tensor(shape, device, dtype, std=0.02):
    t = torch.empty(shape, device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.trunc_normal_(t, mean=0.0, std=std)
    return t
