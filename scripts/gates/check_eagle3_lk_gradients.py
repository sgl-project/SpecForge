"""Check EAGLE3 LK zero-acceptance rows through backward and an optimizer step."""

import argparse
import math

import torch

from specforge.core.lk_loss import compute_acceptance_rate, compute_lk_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    # The supervised row has q=(sigmoid(2), 1-sigmoid(2)) and acceptance=0.8+q1.
    q0 = 1 / (1 + math.exp(-2))
    acceptance = 0.8 + (1 - q0)
    derivative = q0 * (1 - q0) / acceptance
    scenarios = (
        ("masked padding", [1000.0, -1000.0], [0.0, 0.0], [1.0, 0.0]),
        ("disjoint support", [1000.0, -1000.0], [0.0, 1.0], [1.0, 1.0]),
        ("ordinary padding", [1.0, -1.0], [0.0, 0.0], [1.0, 0.0]),
        ("fractional mask", [1000.0, -1000.0], [0.0, 0.0], [0.5, 0.0]),
        ("empty mask", [1000.0, -1000.0], [0.0, 0.0], [0.0, 0.0]),
    )
    for name, zero_row_logits, zero_row_target, mask_values in scenarios:
        logits = torch.nn.Parameter(
            torch.tensor([[[2.0, 0.0], zero_row_logits]], device=device)
        )
        target = torch.tensor([[[0.8, 0.2], zero_row_target]], device=device)
        mask = torch.tensor(mask_values, device=device).reshape(1, 2, 1)
        weight = mask_values[0] / max(sum(mask_values), 1e-8)
        expected_loss = torch.tensor(-math.log(acceptance) * weight, device=device)
        expected_gradient = torch.tensor(
            [[[derivative * weight, -derivative * weight], [0.0, 0.0]]],
            device=device,
        )
        optimizer = torch.optim.SGD([logits], lr=0.1)
        initial_parameters = logits.detach().clone()
        rate, log_rate = compute_acceptance_rate(
            logits=logits, target_probs=target, position_mask=mask
        )
        loss = compute_lk_loss(
            kl_loss=torch.zeros((), device=device),
            acceptance_rate=rate,
            log_acceptance_rate=log_rate,
            lk_loss_type="alpha",
            kl_scale=1.0,
            kl_decay=1.0,
        )
        loss.backward()
        if not torch.allclose(loss, expected_loss, atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"{name}: unexpected loss {loss.detach()}")
        if not torch.allclose(logits.grad, expected_gradient, atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"{name}: unexpected gradients {logits.grad}")
        optimizer.step()
        expected_parameters = initial_parameters - 0.1 * expected_gradient
        if not torch.allclose(logits, expected_parameters, atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"{name}: optimizer state diverged from the gradient")
        optimizer.zero_grad(set_to_none=True)
        _, next_log_rate = compute_acceptance_rate(
            logits=logits, target_probs=target, position_mask=mask
        )
        (-next_log_rate).backward()
        if not bool(torch.isfinite(logits.grad).all()):
            raise RuntimeError(f"{name}: nonfinite gradients on the next step")
        print(f"PASS: {name}")


if __name__ == "__main__":
    main()
