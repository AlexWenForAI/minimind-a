import os
import sys
import time
import argparse
import torch
import warnings
import gradio as gr
import numpy as np
import soundfile as sf
from queue import Queue
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, WhisperProcessor
from model.model_audio import MiniMindAudio, AudioConfig
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    audio_config = AudioConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        max_seq_len=args.max_seq_len, 
        use_moe=bool(args.use_moe),
        audio_special_token="<|audio|>"
    )
    
    if 'model' in args.load_from:
        moe_path = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_path}.pth'
        model = MiniMindAudio(audio_config)
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    
    print(f'Audio-LLM参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    
    # 初始化 Whisper 处理器
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    
    # 绑定音频 ID 到模型参数
    audio_id = tokenizer.convert_tokens_to_ids(audio_config.audio_special_token)
    model.params.audio_ids = [audio_id]
    
    return model.eval().to(args.device), tokenizer, processor

class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)

def process_audio_features(audio_path, processor):
    """读取音频文件并提取特征"""
    audio_array, sr = sf.read(audio_path)
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=-1)
    
    # 转换为 16k 采样率（Whisper 标准）
    if sr != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    
    # 解决环境类型校验问题的 tolist() 策略
    audio_input = audio_array.astype(np.float32).tolist()
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt")
    return inputs.input_features

def chat(prompt, current_audio_path):
    global temperature, top_p
    
    # 1. 提取音频特征
    input_features = process_audio_features(current_audio_path, processor).to(args.device)
    
    # 2. 构造 Prompt（确保包含音频占位符）
    audio_tag = "<|audio|>"
    full_prompt = f"{audio_tag}\n{prompt}"
    messages = [{"role": "user", "content": full_prompt}]

    new_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    with torch.no_grad():
        inputs = tokenizer(new_prompt, return_tensors="pt").to(args.device)
        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            model.generate(
                inputs.input_ids,
                input_features=input_features, # 传入音频特征
                max_new_tokens=args.max_seq_len,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )

        Thread(target=_generate).start()

        while True:
            text = queue.get()
            if text is None: break
            yield text

def launch_gradio_server(server_name="0.0.0.0", server_port=8888):
    global temperature, top_p
    temperature, top_p = args.temperature, args.top_p

    with gr.Blocks(title="MiniMind-Audio") as demo:
        gr.HTML("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <span style="font-size:40px; font-weight:bold; font-style: italic;">Hi, I'm MiniMind-Audio</span>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                audio_input = gr.Audio(type="filepath", label="上传或录制音频")
                
                with gr.Row():
                    temperature_slider = gr.Slider(label="Temperature", minimum=0.1, maximum=1.5, value=0.7)
                    top_p_slider = gr.Slider(label="Top-P", minimum=0.5, maximum=1.0, value=0.9)
                
                def update_params(temp, tp):
                    global temperature, top_p
                    temperature, top_p = temp, tp

                temperature_slider.change(update_params, [temperature_slider, top_p_slider])
                top_p_slider.change(update_params, [temperature_slider, top_p_slider])

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(label="对话历史", height=600)
                msg_input = gr.Textbox(placeholder="询问关于音频的内容...", label="输入问题")
                
                with gr.Row():
                    clear_btn = gr.Button("清空记录")
                    submit_btn = gr.Button("发送", variant="primary")

                def user_chat(message, history, audio_path):
                    if not audio_path:
                        return history + [(message, "请先上传音频文件。")]
                    
                    response = ""
                    history = history + [(message, "")]
                    for text in chat(message, audio_path):
                        response += text
                        history[-1] = (message, response)
                        yield history

                submit_btn.click(user_chat, [msg_input, chatbot, audio_input], chatbot)
                clear_btn.click(lambda: None, None, chatbot, queue=False)

    demo.launch(server_name=server_name, server_port=server_port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_from', default='../model', type=str)
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--weight', default='pretrain_audio', type=str)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=0, type=int)
    args = parser.parse_args()

    model, tokenizer, processor = init_model(args)
    launch_gradio_server()