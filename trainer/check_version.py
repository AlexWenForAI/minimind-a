import sys
import numpy as np
import torch

# 定义需要检查的库列表
libs_to_check = [
    "transformers",
    "tokenizers",
    "torch",
    "numpy",
    "soundfile",  # Whisper 常用依赖
    "librosa",    # Whisper 常用依赖
    "accelerate"  # transformers 训练常用依赖
]

# 打印 Python 版本
print(f"Python 版本: {sys.version}")
print("-" * 50)

# 循环检查并打印各库版本
for lib_name in libs_to_check:
    try:
        lib = __import__(lib_name)
        version = lib.__version__
        print(f"{lib_name:<12} 版本: {version}")
    except ImportError:
        print(f"{lib_name:<12} 未安装")
    except AttributeError:
        print(f"{lib_name:<12} 版本无法获取")

# 额外检查 PyTorch CUDA 可用性（可选）
print("-" * 50)
print(f"PyTorch CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")