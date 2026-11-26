# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def parse_harmony_message_content(content):
    """
    解析 content 字符串中的 Harmony 格式。
    如果匹配到 Harmony 格式，返回包含 channel 和 content 的列表；
    否则，返回原内容并标记为默认 channel。
    """
    # 匹配 <|channel|>xxx<|message|>yyy<|end|>
    pattern = r"<\|channel\|>(.*?)<\|message\|>(.*?)<\|end|>"
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        # 如果没有匹配到 Harmony 标签，视作普通文本
        return [{"channel": "text", "content": content}]

    results = []
    for channel, msg_body in matches:
        results.append({"channel": channel.strip(), "content": msg_body.strip()})
    return results


def process_harmony_conversations(conversation):
    """
    处理传入的 list[list[dict]] 结构
    """
    new_conversation = []
    for msg in conversation:
        role = msg.get("role")
        original_content = msg.get("content", "")

        # 解析 content 中的 Harmony 结构
        segments = parse_harmony_message_content(original_content)

        # 为每个解析出的通道生成一个新的消息字典
        for seg in segments:
            new_msg = {
                "role": role,
                "channel": seg["channel"],  # 新增字段标识通道
                "content": seg["content"],
            }
            new_conversation.append(new_msg)

    return new_conversation
