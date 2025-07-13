from datasets import load_dataset, Dataset
from tqdm import tqdm
import logging

class DataProcessorBase:
    dataset_name = None

    def process(self):
        logging.info(f"Processing {self.dataset_name} dataset...")
        return self._process()

    def _process(self):
        raise NotImplementedError("Subclasses must implement _process()")

class ShareGPTProcessor(DataProcessorBase):
    dataset_name = "Aeala/ShareGPT_Vicuna_unfiltered"

    def _process(self):
        ds = load_dataset(self.dataset_name)
        ROLE_MAPPING = {
            "human": "user",
            "gpt": "assistant"
        }
        
        def process_item(item):
            new_conversations = [
                {
                    "role": ROLE_MAPPING[message['from']],
                    "content": message['value']
                }
                for message in item['conversations']
            ]
            return {
                "id": item["id"],
                "conversations": new_conversations
            }
        
        all_data = [process_item(item) for item in tqdm(ds["train"], desc="Processing ShareGPT")]
        return Dataset.from_list(all_data)

class Ultrachat200KProcessor(DataProcessorBase):
    dataset_name = "HuggingFaceH4/ultrachat_200k"

    def _process(self):
        ds = load_dataset(self.dataset_name)
        
        def process_item(item):
            new_conversations = [
                {
                    "role": message['role'],
                    "content": message['content']
                }
                for message in item['messages']
                if message['role'] in ["user", "assistant"]
            ]
            return {
                "id": item["prompt_id"],
                "conversations": new_conversations
            }
        
        all_data = [process_item(item) for item in tqdm(ds["train_sft"], desc="Processing Ultrachat200K")]
        return Dataset.from_list(all_data)