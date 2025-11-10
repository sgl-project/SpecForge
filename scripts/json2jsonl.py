import json


def convert_to_jsonl(input_file, output_file):
    """将对话格式的 JSON 转换为 JSONL 格式"""

    with open(input_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    i = 0
    f = open(output_file, 'w', encoding='utf-8')
    for data in datas:
        # 构建新的对话结构
        conversations = []

        for message in data["messages"]:
            # 映射角色
            if message["from"] in ["human", "observation"]:
                role = "user"
            elif message["from"] in ["gpt", "function_call"]:
                role = "assistant"
            else:
                continue  # 跳过未知类型

            # 添加到对话列表
            conversations.append({
                "role": role,
                "content": message["value"]
            })

        # 构建最终输出格式
        output_data = {
            "conversations": conversations
        }

        # 写入 JSONL 文件（每行一个 JSON 对象）

        json_line = json.dumps(output_data, ensure_ascii=False)
        f.write(json_line + '\n')
        i += 1
    print(f"转换完成！输出文件: {output_file}")
    print(f"共转换 {i} 条对话记录")


# 使用示例
convert_to_jsonl('/nfs/ofs-llm-ssd/user/daiyajun/project/research/megatron_sft_toolkits/dataset/dtaxi_v31_dt1029.json', '/nfs/ofs-fengyu/data/dtaxi_v31_dt1029.jsonl')
