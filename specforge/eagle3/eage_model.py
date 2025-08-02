from typing import Optional, Union
from .utils import *
import torch
import torch.nn as nn
from .kv_cache import initialize_past_key_values
from time import time


class EaModel(nn.Module):
    def __init__(
            self,
            target_model,
            draft_model,
            tokenizer,
            total_tokens=63, 
            depth=5, 
            top_k=8, 
            threshold=1.0
    ):
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.draft_model.set_params(total_tokens, depth, top_k, threshold)
        self.draft_model.init_tree()
    
    def forward(
            self,
            input_ids=None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            inputs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=True
            )
            outputs = self.target_model.model(
                **inputs
            )
            if output_orig:
                orig = self.target_model.lm_head(outputs[0])
            #hidden_states = outputs[0]
            hidden_states = outputs["hidden_states"]

        if output_orig:
            return outputs, orig, hidden_states, outputs["past_key_values"]
        else:
            return outputs, hidden_states, outputs["past_key_values"]
        
    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            pixel_values,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,

    ):
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        input_ids = input_ids.clone()
        self.draft_model.reset_kv()
        # past_key_values = StaticCache(config=self.target_model.config, max_batch_size=1, max_cache_len=2048,device="cuda",dtype=torch.float16)
        past_key_values = initialize_past_key_values(self)
        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.target_model(input_ids, pixel_values, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        new_token = 0
        max_length = max_length - self.draft_model.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.target_model(input_id, use_cache=True, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values 
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids[0, input_len:]
        else:
            return input_ids, new_token, idx
        
    @torch.no_grad()
    def eagenerate(
        self,
        input_ids,
        pixel_values,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
):
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.draft_model.reset_kv()
        
        past_key_values = initialize_past_key_values(self)
        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        total_time = 0
        start = time()
        # Prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token, past_key_values = initialize_tree(
            input_ids, pixel_values, self, past_key_values, logits_processor
        )
        if log:
            prefill_time = time() - start
            total_time += prefill_time
            print(f"prefill time:{prefill_time}")
        new_token = 0
        max_length = max_length - self.draft_model.total_tokens - 10

        # ---- Initialize for accept_length tracking ----
        total_accept_length = 0
        accept_count = 0
        # -----------------------------------------------

        for idx in range(max_length):
            start = time()
            self.target_model.model.language_model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)
            logits, hidden_state_new, outputs, past_key_values = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            # ---- Update accept_length stats ----
            total_accept_length += accept_length
            accept_count += 1
            # -------------------------------------

            past_key_values_data, current_length_data = past_key_values.get_meta()
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values,
                self,
                hidden_state_new,
                sample_p
            )

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
            if log:
                decode_time = time() - start
                total_time += decode_time
                print(f"decode time:{decode_time}")

        # ---- Print average accept length ----
        average_accept_length = total_accept_length / accept_count if accept_count > 0 else 0
        # print(f"Average accept_length: {average_accept_length:.2f}")
        # --------------------------------------
        if log:
            print(f"total time: {total_time}")

        return input_ids[0, input_len:],average_accept_length
        # else:
        #     return input_ids, new_token, idx,average_accept_length
