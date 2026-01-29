import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
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
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

# 引入新的工具模块
from model.model_audio import AudioConfig
from dataset.audio_dataset import AudioSFTDataset
from trainer.trainer_utils_audio import (
    get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, 
    init_audio_model, audio_checkpoint, SkipBatchSampler
)

warnings.filterwarnings('ignore')

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels, input_features) in enumerate(loader, start=start_step + 1):
        max_id = input_ids.max().item()
        if max_id >= 6401:
            print(f"!!! 警报 !!! 发现非法 Token ID: {max_id}, 模型上限是 6401")
            # 打印出有问题的文本内容，看看是从哪儿来的
            print(tokenizer.decode(input_ids[0]))
            sys.exit(1) # 立刻停止，不触发 CUDA 报错
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        input_features = input_features.to(args.device).to(dtype)
        
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, input_features=input_features)
            loss = res.loss + res.aux_loss
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
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {current_lr:.8f}, time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            # 使用封装好的 audio_checkpoint 保存
            audio_checkpoint(audio_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()

        del input_ids, labels, input_features, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-Audio Pretrain")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain_audio', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=640, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/audio_data/pretrain_audio.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='llm', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1], help="是否冻结LLM参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Audio-Pretrain", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 构造 Config (必须显式传入 token id，但这里我们通过 init_audio_model 内部处理完后可能需要重新更新一下逻辑)
    # 为了简化，我们先创建一个基础 config，后续如果 init_audio_model 加了 token，model 内部会自动适应
    # 注意：MiniMindAudio 初始化时需要 audio_special_token。
    # 我们这里预设好，确保和 init_audio_model 里加的一致。
    audio_special_token = '<|audio|>'
    
    # 这里我们只初始化基础参数，Audio ID 的绑定在 init_audio_model 内部完成加载后可能更合适，
    # 但 MiniMindAudio 需要 config 里的 id 来做占位符识别。
    # 简单的做法：先不传 id，等 tokenizer 加载完再获取。
    # 但 AudioConfig 是 dataclass。
    # 策略：init_audio_model 会处理 tokenizer 和 model 的初始化。
    
    audio_config = AudioConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        max_seq_len=args.max_seq_len, 
        use_moe=bool(args.use_moe),
        audio_special_token=audio_special_token
    )
    
    ckp_data = audio_checkpoint(audio_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb.init(project=args.wandb_project, id=wandb_id, resume=resume, config=args.__dict__)
    else:
        wandb = None
    # ========== 2. 初始化模型、Tokenizer、Processor (一行搞定) ==========
    # 注意：init_audio_model 会负责加载权重、冻结层、Resize Embedding
    model, tokenizer, audio_processor = init_audio_model(
        audio_config, 
        from_weight=args.from_weight, 
        tokenizer_path='../model', 
        audio_model_path="openai/whisper-tiny",
        save_dir='../out',
        device=args.device,
        freeze_llm=bool(args.freeze_llm)
    )
    # --- 强制对齐开始 ---
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id >= 6401:
        # 强制将 pad 指向 <|endoftext|>，对应的 ID 应该是 0
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")  
    print(f"确认当前 Pad ID: {tokenizer.pad_token_id}") # 必须确保输出是 0 到 6400 之间的数
# --- 强制对齐结束 ---
    # 重要：将 tokenizer 里生成的 audio token id 回填给 config
    # 这样模型在 forward 时才能正确识别 <|audio|>
    audio_token_id = tokenizer.convert_tokens_to_ids(audio_special_token)
    model.params.audio_ids = [audio_token_id] # 更新模型内的配置
    
    if args.from_resume == 0 and args.from_weight == 'none':
        Logger("警告: 未加载任何权重，模型将随机初始化！")

    # ========== 3. 数据集与训练 ==========
    train_ds = AudioSFTDataset(
        args.data_path, 
        tokenizer, 
        audio_processor=audio_processor, 
        max_length=audio_config.max_seq_len
    )
    
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 优化器
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # 恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step...')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    if dist.is_initialized(): dist.destroy_process_group()