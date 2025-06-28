# this file is copied from BaldEagle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from transformers import Trainer
from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup_and_decay(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0, last_epoch=-1):
    """
    Create a schedule with linear warmup and then linear decay.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: The number of steps for the warmup phase
        num_training_steps: The total number of training steps
        min_lr_ratio: The final learning rate ratio (min_lr = lr * min_lr_ratio)
        last_epoch: The index of the last epoch when resuming training
    
    Returns:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule
    """
    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Linear decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def top_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, -1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res

class OfflineEagleTrainTimeTestTrainer(Trainer):

    def __init__(self, head, min_lr_ratio=0.0, **kwargs):
        super().__init__(**kwargs)
        self.cross_entropy_loss_1_total = 0
        self.cross_entropy_loss_2_total = 0
        self.cross_entropy_loss_3_total = 0
        self.cross_entropy_loss_4_total = 0
        self.top_1_acc_sum = 0
        self.top_2_acc_sum = 0
        self.top_3_acc_sum = 0
        self.top_4_acc_sum = 0
        self.logging_valid_token_count = 0
        self.steps_since_last_logging = 0
        self.head = head
        self.min_lr_ratio = min_lr_ratio
        self.loss_gamma = 0.5

    def create_scheduler(self, num_training_steps, optimizer=None):
        """
        Override to create a custom linear decay scheduler with warmup.
        """
        if optimizer is None:
            optimizer = self.optimizer
        
        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
        
        self.lr_scheduler = get_linear_schedule_with_warmup_and_decay(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=self.min_lr_ratio
        )

        print(f"Created custom LR scheduler with min_lr_ratio={self.min_lr_ratio}")
        
        return self.lr_scheduler

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        input_ids = inputs["input_ids"]
        hidden_states = inputs["hidden_states"]
        attention_mask = inputs["attention_mask"]
        target_hidden_states = inputs["target"]
        loss_mask = inputs["loss_mask"][:, :, None]

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            with torch.no_grad():
                predicted_hidden_states = model(input_ids=input_ids,
                                                hidden_state=hidden_states,
                                                attention_mask=attention_mask)
                
                predicted_hidden_states_1 = model(input_ids=input_ids[:, 1:],
                                                  hidden_state=predicted_hidden_states[:, :-1],
                                                  attention_mask=attention_mask[:, 1:])

                predicted_hidden_states_2 = model(input_ids=input_ids[:, 2:],
                                                  hidden_state=predicted_hidden_states_1[:, :-1],
                                                  attention_mask=attention_mask[:, 2:])
                
                predicted_hidden_states_3 = model(input_ids=input_ids[:, 3:],
                                                  hidden_state=predicted_hidden_states_2[:, :-1],
                                                  attention_mask=attention_mask[:, 3:])
                        
                target_head = self.head(target_hidden_states)
                target_probs = nn.Softmax(dim=2)(target_head)
                _, target = torch.max(target_head, 2)

                pred_head_1 = self.head(predicted_hidden_states)
                pred_log_probs = nn.LogSoftmax(dim=2)(pred_head_1)
                loss_class_1 = target_probs * pred_log_probs
                loss_class_1 = -torch.sum(torch.sum(loss_mask * loss_class_1, 2)) / (loss_mask.sum() + 1e-5)

                pred_head_2 = self.head(predicted_hidden_states_1)
                pred_log_probs_2 = nn.LogSoftmax(dim=2)(pred_head_2)
                loss_class_2 = target_probs[:, 1:] * pred_log_probs_2
                loss_class_2 = -torch.sum(torch.sum(loss_mask[:, 1:] * loss_class_2, 2)) / (loss_mask[:, 1:].sum() + 1e-5)

                pred_head_3 = self.head(predicted_hidden_states_2)
                pred_log_probs_3 = nn.LogSoftmax(dim=2)(pred_head_3)
                loss_class_3 = target_probs[:, 2:] * pred_log_probs_3
                loss_class_3 = -torch.sum(torch.sum(loss_mask[:, 2:] * loss_class_3, 2)) / (loss_mask[:, 2:].sum() + 1e-5)

                pred_head_4 = self.head(predicted_hidden_states_3)
                pred_log_probs_4 = nn.LogSoftmax(dim=2)(pred_head_4)
                loss_class_4 = target_probs[:, 3:] * pred_log_probs_4
                loss_class_4 = -torch.sum(torch.sum(loss_mask[:, 3:] * loss_class_4, 2)) / (loss_mask[:, 3:].sum() + 1e-5)

                total_loss = (loss_class_1 + loss_class_2 + loss_class_3 + loss_class_4) / 4

                if prediction_loss_only:
                    return (total_loss, None, None)
                return (total_loss, pred_head_1, (target_head, loss_mask, loss_class_1, loss_class_2, loss_class_3, loss_class_4))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):       
        input_ids = inputs["input_ids"]
        hidden_states = inputs["hidden_states"]
        attention_mask = inputs["attention_mask"]
        target_hidden_states = inputs["target"]
        loss_mask = inputs["loss_mask"][:, :, None]
        
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # Forward pass in one go.

            predicted_hidden_states = model(input_ids=input_ids,
                                            hidden_state=hidden_states,
                                            attention_mask=attention_mask)
            
            predicted_hidden_states_1 = model(input_ids=input_ids[:, 1:],
                                              hidden_state=predicted_hidden_states[:, :-1],
                                              attention_mask=attention_mask[:, 1:])
        
            predicted_hidden_states_2 = model(input_ids=input_ids[:, 2:],
                                                hidden_state=predicted_hidden_states_1[:, :-1],
                                                attention_mask=attention_mask[:, 2:])
            
            predicted_hidden_states_3 = model(input_ids=input_ids[:, 3:],
                                              hidden_state=predicted_hidden_states_2[:, :-1],
                                              attention_mask=attention_mask[:, 3:])
            
            # Currently thinking (all for seq len level changes)
            # 1. remove index 0 from input_ids, attention_mask, and loss_mask
            # 2. remove index -1 from predicted_hidden_states
            # 3. run model again on input_ids, attention_mask, predicted_hidden_states, and compute loss with loss_mask
        
            with torch.no_grad():
                target_head = self.head(target_hidden_states)
                target_probs = nn.Softmax(dim=2)(target_head)
            
            pred_head = self.head(predicted_hidden_states)
            pred_log_probs = nn.LogSoftmax(dim=2)(pred_head)
            loss_class_0 = target_probs * pred_log_probs
            loss_class_0 = -torch.sum(torch.sum(loss_mask * loss_class_0, 2)) / (loss_mask.sum() + 1e-5)
            
            pred_head_1 = self.head(predicted_hidden_states_1)
            pred_log_probs_1 = nn.LogSoftmax(dim=2)(pred_head_1)
            loss_class_1 = target_probs[:, 1:] * pred_log_probs_1
            loss_class_1 = -torch.sum(torch.sum(loss_mask[:, 1:] * loss_class_1, 2)) / (loss_mask[:, 1:].sum() + 1e-5)

            pred_head_2 = self.head(predicted_hidden_states_2)
            pred_log_probs_2 = nn.LogSoftmax(dim=2)(pred_head_2)
            loss_class_2 = target_probs[:, 2:] * pred_log_probs_2
            loss_class_2 = -torch.sum(torch.sum(loss_mask[:, 2:] * loss_class_2, 2)) / (loss_mask[:, 2:].sum() + 1e-5)

            pred_head_3 = self.head(predicted_hidden_states_3)
            pred_log_probs_3 = nn.LogSoftmax(dim=2)(pred_head_3)
            loss_class_3 = target_probs[:, 3:] * pred_log_probs_3
            loss_class_3 = -torch.sum(torch.sum(loss_mask[:, 3:] * loss_class_3, 2)) / (loss_mask[:, 3:].sum() + 1e-5)

            total_loss = (loss_class_0 + loss_class_1 + loss_class_2 + loss_class_3) / 4 / self.args.gradient_accumulation_steps

        with torch.no_grad():
            _, target = torch.max(target_head, 2)
            pred_head = pred_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(pred_head, target, (1, 2, 3))

            valid_tokens = loss_mask.sum().item()
            self.cross_entropy_loss_1_total += loss_class_0.item()
            self.cross_entropy_loss_2_total += loss_class_1.item()
            self.cross_entropy_loss_3_total += loss_class_2.item()
            self.cross_entropy_loss_4_total += loss_class_3.item()
            self.top_1_acc_sum += topkacc[0]
            self.top_2_acc_sum += topkacc[1]
            self.top_3_acc_sum += topkacc[2]
            self.logging_valid_token_count += valid_tokens

        if model.training:
            self.steps_since_last_logging += 1

        return total_loss
    
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None):

        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:

            self.log({
                # "regression_loss": self.regression_loss_total / self.steps_since_last_logging,
                # "classification_loss": self.classification_loss_total / self.steps_since_last_logging,
                "cross_entropy_loss_1": self.cross_entropy_loss_1_total / self.steps_since_last_logging,
                "cross_entropy_loss_2": self.cross_entropy_loss_2_total / self.steps_since_last_logging,
                "cross_entropy_loss_3": self.cross_entropy_loss_3_total / self.steps_since_last_logging,
                "cross_entropy_loss_4": self.cross_entropy_loss_4_total / self.steps_since_last_logging,
                "top_1_acc": self.top_1_acc_sum / self.logging_valid_token_count,
                "top_2_acc": self.top_2_acc_sum / self.logging_valid_token_count,
                "top_3_acc": self.top_3_acc_sum / self.logging_valid_token_count,
            })
            
            self.cross_entropy_loss_1_total = 0
            self.cross_entropy_loss_2_total = 0
            self.cross_entropy_loss_3_total = 0
            self.cross_entropy_loss_4_total = 0
            self.top_1_acc_sum = 0
            self.top_2_acc_sum = 0
            self.top_3_acc_sum = 0
            self.steps_since_last_logging = 0
            self.logging_valid_token_count = 0

            self.control.should_log = True
        
        super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate)
    
    def evaluate(self, eval_dataset=None, ignore_keys=False):
        eval_dataset = eval_dataset or self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
    
        top_1_acc_sum, top_2_acc_sum, top_3_acc_sum, total_rows, total_valid_tokens = 0, 0, 0, 0, 0
        losses = []
        losses_class_1 = []
        losses_class_2 = []
        losses_class_3 = []
        losses_class_4 = []
        with torch.no_grad():
            for batch in eval_dataloader:
                loss, logits, labels = self.prediction_step(
                    self.model, batch, prediction_loss_only=False, ignore_keys=ignore_keys
                )
                target_head, loss_mask, loss_class_1, loss_class_2, loss_class_3, loss_class_4 = labels

                losses.append(loss.cpu())
                losses_class_1.append(loss_class_1.cpu())
                losses_class_2.append(loss_class_2.cpu())
                losses_class_3.append(loss_class_3.cpu())
                losses_class_4.append(loss_class_4.cpu())

                _, target = torch.max(target_head, 2)
                pred_head = logits.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(pred_head, target, (1, 2, 3))

                valid_tokens = loss_mask.sum().item()
                top_1_acc_sum += topkacc[0]
                top_2_acc_sum += topkacc[1]
                top_3_acc_sum += topkacc[2]
                total_valid_tokens += valid_tokens
                # TODO: this only assumes bs=1, match bs
                total_rows += 1
        
        metrics = {
            "eval_loss": np.mean(losses).item(),
            "eval_loss_class_1": np.mean(losses_class_1).item(),
            "eval_loss_class_2": np.mean(losses_class_2).item(),
            "eval_loss_class_3": np.mean(losses_class_3).item(),
            "eval_loss_class_4": np.mean(losses_class_4).item(),
            "eval_top_1_acc": top_1_acc_sum  / total_valid_tokens,
            "eval_top_2_acc": top_2_acc_sum / total_valid_tokens,
            "eval_top_3_acc": top_3_acc_sum / total_valid_tokens,
        }

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        return metrics
