import os
from huggingface_hub import snapshot_download

def download_model(model_id, local_dir):
    """
    使用 huggingface_hub 的 snapshot_download 函数下载整个模型仓库。
    """
    print(f"开始下载模型: {model_id}")
    print(f"将保存到: {local_dir}")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 建议False以避免符号链接问题
            # 如果需要下载特定版本，可以取消注释下面这行
            # revision="main"
        )
        print("模型下载成功！")
    except Exception as e:
        print(f"下载过程中发生错误: {e}")

if __name__ == "__main__":
    # 定义模型ID和本地存储目录
    model_identifier = "llama-duo/llama3.1-8b-summarize-gpt4o-128k"
    save_directory = f"./{model_identifier.replace('/', '_')}" # 将'/'替换为'_'以创建有效的文件夹名

    # 创建目标文件夹（如果不存在）
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    download_model(model_id=model_identifier, local_dir=save_directory)