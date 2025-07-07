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
        all_data = []
        for item in tqdm(ds["train"]):
            conversations = item['conversations']
            new_conversations = []
            for message in conversations:
                new_role = ROLE_MAPPING[message['from']]
                content = message['value']
                new_conversations.append({
                    "role": new_role,
                    "content": content
                })
            row = {
                "id": item["id"],
                "conversations": new_conversations
            }
            all_data.append(row)
        return Dataset.from_list(all_data)

class Ultrachat200KProcessor(DataProcessorBase):
    dataset_name = "HuggingFaceH4/ultrachat_200k"

    def _process(self):
        ds = load_dataset(self.dataset_name)
        all_data = []
        for item in tqdm(ds["train_sft"]):
            conversations = item['messages']
            new_conversations = []
            for message in conversations:
                role = message['role']
                content = message['content']
                assert role in ["user", "assistant"]
                new_conversations.append({
                    "role": role,
                    "content": content
                })
            row = {
                "id": item["prompt_id"],
                "conversations": new_conversations
            }
            all_data.append(row)
        return Dataset.from_list(all_data)