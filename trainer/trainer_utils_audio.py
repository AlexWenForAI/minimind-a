"""
Audio 训练工具函数集合
(基于 trainer_utils_vlm 修改适配 MiniMind-Audio)
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, WhisperProcessor, LlamaTokenizer
from model.model_audio import MiniMindAudio
from transformers import PreTrainedTokenizerFast
import json
def get_model_params(model, config, ignore_patterns=['audio_encoder']):
    def should_count(n): return not any(p in n for p in ignore_patterns)
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n and should_count(n)) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n and should_count(n)) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)

def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))

def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_audio_model(audio_config, from_weight='pretrain_audio', tokenizer_path='../model', 
                   audio_model_path='openai/whisper-tiny', 
                   save_dir='../out', device='cuda', freeze_llm=False):
    
    # ================= 核心修改 =================
    # 改回使用本地路径 (tokenizer_path)，因为你刚才已经运行脚本修复了它！
    # 不要再用 "jingyaogong/minimind..." 了
    print(f"正在加载本地 Tokenizer: {tokenizer_path} ...")

    try:
#         # 因为我们修复了 JSON 格式，现在可以直接用 AutoTokenizer + use_fast=True
#         tokenizer = PreTrainedTokenizerFast(
#         tokenizer_file=f"{tokenizer_path}/tokenizer.json",  # 你的自定义词表文件
#         tokenizer_config_file=f"{tokenizer_path}/tokenizer_config.json",
#         vocab_size=6400,  # 强制绑定 MiniMind 6400 词表
#         local_files_only=True,  # 不联网，仅用本地文件
#         use_fast=True,  # 适配自定义词表的 fast 模式
#         padding_side="right",  # 匹配 MiniMind 训练配置
#         truncation_side="right"
#     )
#         special_tokens_dict = {
#     'bos_token': "<|im_start|>",
#     'eos_token': "<|im_end|>",
#     'pad_token': "<|endoftext|>",
#     'unk_token': "<|endoftext|>"
# }
#         tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True, trust_remote_code=True)
        print("本地 AutoTokenizer 加载成功！")
    except Exception as e:
        print(f"本地 AutoTokenizer 加载失败: {e}")
        # 如果本地还是不行，最后一次尝试 Qwen (它的格式是标准的)
        print("尝试使用兼容的 Qwen Tokenizer 作为替代...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", use_fast=True, trust_remote_code=True)
        
    # 2. 关键：确保 <|audio|> token 存在
    audio_special_token = '<|audio|>'
    # 检查是否已有该 token
    if audio_special_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': [audio_special_token]})
        Logger(f"Added special token: {audio_special_token}")
        
    # 手动补全 Chat Template (防止报错)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    
    # 补全 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 初始化模型
    model = MiniMindAudio(audio_config, audio_model_path=audio_model_path)
    
    # 4. 关键：Resize Embedding 以适应新 Token
    model.resize_token_embeddings(len(tokenizer))
    
    # 5. 加载权重
    if from_weight != 'none':
        moe_suffix = '_moe' if audio_config.use_moe else ''
        #weight_path = f'{save_dir}/{from_weight}_{audio_config.hidden_size}{moe_suffix}.pth'
        weight_path = "/root/minimind/out/model_512.pth"
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location=device)
            
            # 1. 探测权重文件中的词表大小 (支持 MiniMind 不同版本的 key 命名)
            embed_key = 'model.embed_tokens.weight' if 'model.embed_tokens.weight' in state_dict else 'tok_embeddings.weight'
            if embed_key in state_dict:
                checkpoint_vocab_size = state_dict[embed_key].shape[0] # 比如 6400 或 151666
            else:
                checkpoint_vocab_size = 0
                Logger(f"Warning: Cannot find embedding weights in {weight_path}")

            # 2. 获取当前模型（已加过 <|audio|>）的真实词表大小
            current_model_vocab_size = model.get_input_embeddings().weight.shape[0] # 这里是 6401

            if checkpoint_vocab_size != 0 and checkpoint_vocab_size != current_model_vocab_size:
                Logger(f"检测到词表不一致: 权重({checkpoint_vocab_size}) vs 模型({current_model_vocab_size})")

                # 情况 A: 权重是巨大的 Qwen 词表 (151666)，这种必须强制扩容模型
                if checkpoint_vocab_size > 10000:
                    Logger(f"检测到权重属于大词表系统，正在强制扩容模型维度以匹配权重...")
                    model.model.embed_tokens = torch.nn.Embedding(checkpoint_vocab_size, model.params.hidden_size)
                    model.lm_head = torch.nn.Linear(model.params.hidden_size, checkpoint_vocab_size, bias=False)
                    # 这种情况下需要 strict=False 加载
                    model.load_state_dict(state_dict, strict=False)
                
                # 情况 B: 权重是 6400，模型是 6401 (多了一个 <|audio|>)
                # 这种绝对不能 resize 模型！我们要保持 6401，让 load_state_dict 自动忽略不匹配的部分
                elif checkpoint_vocab_size < current_model_vocab_size:
                    Logger(f"保持模型当前维度 {current_model_vocab_size}，仅加载前 {checkpoint_vocab_size} 个 Token 的权重...")
                    # 移除 state_dict 中尺寸不符的 key，防止 load_state_dict 报错
                    state_dict.pop('model.embed_tokens.weight', None)
                    state_dict.pop('tok_embeddings.weight', None)
                    state_dict.pop('lm_head.weight', None)
                    state_dict.pop('output.weight', None)
                    # 剩下的权重 (Transformer 层等) 会正常加载
                    model.load_state_dict(state_dict, strict=False)
            else:
                # 词表一致，正常加载
                model.load_state_dict(state_dict, strict=False)

            Logger(f"Loaded weights from {weight_path}")
        else:
            Logger(f"Weight path {weight_path} not found, using random initialization.")

    # 6. Pretrain阶段：冻结除 audio_proj 外的所有参数
    if freeze_llm:
        Logger("冻结 LLM 参数，仅训练 Audio Projector")
        for name, param in model.named_parameters():
            if 'audio_proj' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    # 7. 打印参数统计
    get_model_params(model, audio_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    
    # 8. 获取 Processor
    preprocess = model.processor
    
    return model.to(device), tokenizer, preprocess

def audio_checkpoint(audio_config, weight='pretrain_audio', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if audio_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{audio_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{audio_config.hidden_size}{moe_path}_resume.pth'
    
    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        
        # 移除 audio_encoder 参数（不需要保存，因为是预训练的 Whisper）
        clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('audio_encoder.')}
        
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
        
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, clean_state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
    
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)