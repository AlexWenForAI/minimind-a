import time
import argparse
import os
import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
import warnings
import torch
import soundfile as sf
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, WhisperProcessor
from model.model_audio import MiniMindAudio, AudioConfig
from trainer.trainer_utils_audio import setup_seed, init_audio_model

warnings.filterwarnings('ignore')

def process_audio(audio_path, processor, target_sr=16000):
    """鲁棒的音频处理：读取 -> 重采样 -> 特征提取"""
    try:
        audio_array, native_sr = sf.read(audio_path)
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=-1)
        
        # 强制重采样
        if native_sr != target_sr:
            from scipy.signal import resample
            num_samples = int(len(audio_array) * target_sr / native_sr)
            audio_array = resample(audio_array, num_samples).astype(np.float32)
        
        # 使用 tolist() 彻底规避 "expected np.ndarray" 的环境 Bug
        audio_input = audio_array.tolist()
        inputs = processor(audio_input, sampling_rate=target_sr, return_tensors="pt")
        return inputs.input_features.to(torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    except Exception as e:
        print(f"音频处理失败 {audio_path}: {e}")
        return None

def init_eval_model(args):
    """借鉴 Pretrain 逻辑的初始化函数"""
    audio_special_token = '<|audio|>'
    
    # 1. 构造与训练一致的 Config
    audio_config = AudioConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        max_seq_len=args.max_seq_len, 
        use_moe=bool(args.use_moe),
        audio_special_token=audio_special_token
    )
    
    # 2. 调用 trainer_utils_audio 中的 init_audio_model
    # 该函数内部会自动处理：本地 Tokenizer 失败 -> 回退到 Qwen -> Add Token -> Resize Embedding
    model, tokenizer, audio_processor = init_audio_model(
        audio_config, 
        from_weight='model',  # 强制从本地 .pth 加载
        tokenizer_path='../model', 
        audio_model_path="openai/whisper-tiny",
        save_dir=args.save_dir,
        device=args.device,
        freeze_llm=False # 推理不需要冻结
    )
    
    # 3. 核心：加载具体的权重文件
    # 假设权重名为 pretrain_audio.pth 或 sft_audio.pth
    ckp_path = os.path.join(args.save_dir, f"{args.weight}.pth")
    if os.path.exists(ckp_path):
        print(f"正在加载权重: {ckp_path}")
        state_dict = torch.load(ckp_path, map_location=args.device)
        # 如果是 trainer 保存的 checkpoint，可能在 'model' 键下
        if 'model' in state_dict: state_dict = state_dict['model']
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:
        print(f"警告：未找到权重文件 {ckp_path}，将使用初始化权重进行测试。")

    # 4. 绑定音频 ID (与 Pretrain 逻辑完全一致)
    audio_token_id = tokenizer.convert_tokens_to_ids(audio_special_token)
    model.params.audio_ids = [audio_token_id]
    
    return model.eval().to(args.device), tokenizer, audio_processor

def main():
    parser = argparse.ArgumentParser(description="MiniMind-Audio Chat")
    parser.add_argument('--save_dir', default='../out', type=str, help="权重目录")
    parser.add_argument('--weight', default='pretrain_audio', type=str, help="权重文件名")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=0, type=int)
    parser.add_argument('--audio_dir', default='../dataset/eval_audios/', type=str)
    args = parser.parse_args()

    model, tokenizer, processor = init_eval_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 对话模板：确保包含 <|audio|>
    audio_tag = '<|audio|>'
    prompt_template = f"{audio_tag}描述一下这段音频的内容。"

    for audio_file in sorted(os.listdir(args.audio_dir)):
        if audio_file.lower().endswith(('.flac', '.wav', '.mp3')):
            setup_seed(42)
            audio_path = os.path.join(args.audio_dir, audio_file)
            
            input_features = process_audio(audio_path, processor)
            if input_features is None: continue
            input_features = input_features.to(args.device).to(model.dtype)

            messages = [{"role": "user", "content": prompt_template}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer(input_text, return_tensors="pt").to(args.device)

            print(f'\n[音频测试]: {audio_file}')
            print(f'🤖: ', end='')
            
            with torch.no_grad():
                model.generate(
                    inputs=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    input_features=input_features,
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=0.85,
                    temperature=0.65,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer
                )
            print("-" * 30)

if __name__ == "__main__":
    main()