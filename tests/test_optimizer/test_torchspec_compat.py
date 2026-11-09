import copy
import math
import unittest
from unittest.mock import patch
import torch
from specforge.config.schema import TrainingConfig
from specforge.optimizer import BF16Optimizer

class TorchSpecCompatibility(unittest.TestCase):
    def make(self):
        model=torch.nn.Linear(2,2,bias=False).to(torch.bfloat16)
        opt=BF16Optimizer(model,lr=1e-4,weight_decay=.01,min_lr=1e-5,total_steps=1197,warmup_ratio=.01,lr_scheduler='torchspec_cosine')
        return model,opt

    def test_original_schedule_and_decay(self):
        model,opt=self.make()
        self.assertEqual(opt.optimizer.param_groups[0]['weight_decay'],.01)
        self.assertEqual(opt.get_learning_rate(),0)
        for step in range(1,1198):
            model.weight.grad=torch.ones_like(model.weight)
            opt.step()
            expected=1e-4*step/11 if step<=11 else 1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*(step-11)/(1197-11)))
            self.assertAlmostEqual(opt.get_learning_rate(),expected,places=14)

    def test_checkpoint_next_update(self):
        model,opt=self.make()
        for _ in range(12):
            model.weight.grad=torch.ones_like(model.weight);opt.step()
        other,restored=self.make()
        other.load_state_dict(model.state_dict())
        with patch("specforge.optimizer.print_on_rank0"):
            restored.load_state_dict(copy.deepcopy(opt.state_dict()))
        model.weight.grad=torch.full_like(model.weight,.25)
        other.weight.grad=model.weight.grad.clone()
        opt.step();restored.step()
        self.assertEqual(opt.get_learning_rate(),restored.get_learning_rate())
        self.assertTrue(torch.equal(model.weight,other.weight))
        for a,b in zip(opt.fp32_params,restored.fp32_params):self.assertTrue(torch.equal(a,b))

    def test_config_rejects_invalid_min_lr(self):
        for kw in [dict(learning_rate=1e-4,min_lr=.1),dict(lr_scheduler='constant',min_lr=1e-5)]:
            with self.assertRaises(ValueError):TrainingConfig(**kw)
