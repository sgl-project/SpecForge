import os

from huggingface_hub import snapshot_download


def download_model(model_id, local_dir):
    print(f"downloading model: {model_id}")
    print(f"will save to: {local_dir}")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print("download success！")
    except Exception as e:
        print(f"error: {e}")


if __name__ == "__main__":
    model_identifier = "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
    save_directory = f"./{model_identifier.replace('/', '_')}"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    download_model(model_id=model_identifier, local_dir=save_directory)
