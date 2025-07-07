from sgl_eagle.data.data_pipeline import prepare_full_dataloaders

from transformers import AutoTokenizer


model_path="/data/eagle_data/shenggui/models/Llama-4-Scout-17B-16E-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

train_loader, test_loader, train_sampler, test_sampler, target_dict = \
    prepare_full_dataloaders(tokenizer, temp_dir="/data/eagle_data/chao")


