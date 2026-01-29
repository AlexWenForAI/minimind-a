import json
import os

# 定义路径
tokenizer_path = "../model/tokenizer.json"

# 1. 读取原始文件
print(f"正在读取 {tokenizer_path} ...")
with open(tokenizer_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 动手术：给 model 部分添加 type 字段
# MiniMind/Llama 是基于 BPE 的
if "model" in data:
    if "type" not in data["model"]:
        print("发现缺失 'type' 字段，正在修复...")
        data["model"]["type"] = "BPE"  # 核心修复点
    else:
        print(f"当前 type 为: {data['model']['type']}，无需修复。")
else:
    print("错误：JSON 中没有 model 字段！")

# 3. 保存回文件
with open(tokenizer_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("修复完成！现在可以用 AutoTokenizer 加载了。")