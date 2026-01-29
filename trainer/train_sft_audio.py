import os
import sys

__package__ = "trainer"
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.audio_dataset import AudioSFTDataset
# 替换为 Audio 相关的模型和配置
from model.model_audio import MiniMindAudio, AudioConfig
#from dataset.lm_dataset import AudioDataset  # 确保你的 dataset 目录下有 AudioDataset
from trainer.trainer_utils_audio import (
    get_lr, Logger, is_main_process, init_distributed_mode, 
    setup_seed, init_audio_model, audio_checkpoint, SkipBatchSampler
)

warnings.filterwarnings('ignore')

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    # pixel_values 替换为 audio_tensors
    for step, (input_ids, labels, audio_tensors) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        audio_tensors = audio_tensors.to(args.device)
        
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 模型输入改为 audio_tensors
            res = model(input_ids, labels=labels, audio_tensors=audio_tensors)
            loss = res.loss + (res.aux_loss if res.aux_loss is not None else 0.0)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "learning_rate": current_lr})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if audio_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{audio_config.hidden_size}{moe_suffix}.pth'
            
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            
            # 关键：SFT 阶段我们保存 LLM + Projector，但通常不保存冻结的 Whisper Encoder
            clean_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith('audio_encoder.')
            }
            clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
            torch.save(clean_state_dict, ckp)
            
            audio_checkpoint(audio_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                           epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict, clean_state_dict

        del input_ids, labels, audio_tensors, res, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-Audio SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='sft_audio', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=4, help="SFT通常需要更多轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="SFT建议学习率稍微调大一点")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度")
    parser.add_argument("--num_workers", type=int, default=8, help="线程数")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="梯度累积")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--log_interval", type=int, default=10, help="日志间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--use_moe', default=0, type=int)
    # SFT 数据路径
    default_data_path = os.path.abspath(os.path.join(root_path, "dataset", "audio_data", "sft_audio_360.jsonl"))
    parser.add_argument("--data_path", type=str, default="../dataset/audio_data/sft_audio_360.jsonl", help="SFT数据路径")
    # 关键：加载预训练好的 Projector 权重
    parser.add_argument('--from_weight', default='pretrain_audio', type=str, help="加载预训练的音频权重")
    parser.add_argument('--from_resume', default=0, type=int)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    # 1. 环境初始化
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # 2. 配置模型
    os.makedirs(args.save_dir, exist_ok=True)
    audio_config = AudioConfig(hidden_size=args.hidden_size, max_seq_len=args.max_seq_len, use_moe=bool(args.use_moe))
    ckp_data = audio_checkpoint(audio_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # 3. 混合精度
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    
    # 4. Wandb
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project="MiniMind-Audio-SFT", name=f"SFT-LR-{args.learning_rate}")
    
    # 5. 模型、Tokenizer、数据
    # 注意：在 SFT 阶段，init_audio_model 内部应设置 freeze_llm=False
    model, tokenizer, audio_processor = init_audio_model(
        audio_config, 
        from_weight=args.from_weight, 
        device=args.device,
        freeze_llm=False  # SFT 必须放开 LLM 训练
    )
    # --- 核心修复：强制词表对齐并验证 (在此处插入) ---
    # 确保 Tokenizer 包含音频标签，且 ID 为 6400
    if '<|audio|>' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>']})
    
    audio_id = tokenizer.convert_tokens_to_ids('<|audio|>')
    
    # 强制 Resize 模型 Embedding 层到 6401 (0-6400)
    if model.get_input_embeddings().weight.shape[0] != 6401:
        Logger(f"检测到模型维度不足，正在从 {model.get_input_embeddings().weight.shape[0]} 强制调整为 6401...")
        model.resize_token_embeddings(6401)
        # 更新模型内部参数，确保识别占位符
        model.params.audio_ids = [audio_id]
        if hasattr(model, 'module'): # 处理 DDP 情况
            model.module.params.audio_ids = [audio_id]

    # 打印确认
    Logger(f"SFT 启动确认: 词表大小={model.get_input_embeddings().weight.shape[0]}, <|audio|> ID={audio_id}")
    # --- 修复结束 ---
    train_ds =  AudioSFTDataset(
        args.data_path, 
        tokenizer, 
        audio_processor=audio_processor,
        max_length=args.max_seq_len
    )
    
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 6. 恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        start_epoch = ckp_data['epoch']
    
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # 7. 训练循环
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        # SFT 通常不需要复杂的 SkipBatch，这里简化处理
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, 
                          shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True)
        
        train_epoch(epoch, loader, len(loader), 0, wandb)
    
    if dist.is_initialized(): dist.destroy_process_group()