import os
import json
from utils.jsonl_utils import steps_finish_reason
from config_schema import AppConfig

def get_latest_entries_by_prompt_id(input_file_path: str) -> list:
    """
    读取一个JSONL文件，并为每个'prompt_id'只保留其最后一次出现的行。

    Args:
        input_file_path (str): 输入的JSONL文件路径。

    Returns:
        list: 一个包含每个prompt_id最新条目的字典列表。
    """
    if not os.path.exists(input_file_path):
        print(f"错误：文件 '{input_file_path}' 不存在。")
        return []

    # 使用字典来存储每个prompt_id对应的最新数据
    # 键是 prompt_id, 值是整行的json对象（已解析为python字典）
    latest_entries = {}

    print(f"正在处理文件: {input_file_path}...")

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # 忽略空行
                line = line.strip()
                if not line:
                    continue

                try:
                    # 将每一行从JSON字符串解析为Python字典
                    data = json.loads(line)
                    
                    # 获取 prompt_id
                    prompt_id = data.get('prompt_id')

                    if prompt_id is not None:
                        # 如果这个prompt_id已经存在于字典中，这行新的数据会覆盖旧的
                        # 因为我们是按顺序读取文件，所以这自然会保留最后一次出现的行
                        latest_entries[prompt_id] = data
                    else:
                        print(f"警告: 第 {i+1} 行没有找到 'prompt_id' 字段，已跳过。")

                except json.JSONDecodeError:
                    print(f"警告: 第 {i+1} 行不是有效的JSON格式，已跳过。行内容: '{line}'")

    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []

    # 字典的值就是我们想要的最新条目列表
    return list(latest_entries.values())


def read_jsonl(data_file):
    if not os.path.exists(data_file):
        return []
    with open(data_file, 'r') as f:
        data = [json.loads(l.strip()) for l in f.readlines()]
    return data


def write_jsonl(data, data_file):
    with open(data_file, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

