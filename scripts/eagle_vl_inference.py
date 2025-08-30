#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eagle 2.5 视觉语言模型推理脚本
用于图文生成任务，使用项目中的预训练模型参数

功能：
1. 图像描述生成
2. 视觉问答
3. 图像理解对话
4. 支持批量处理

作者：AI Assistant
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from gr00t.model.backbone.eagle2_hg_model import (
    Eagle2_5_VLConfig,
    Eagle2_5_VLForConditionalGeneration,
    Eagle2_5_VLProcessor,
)


class EagleVLInference:
    """Eagle 2.5 视觉语言模型推理类"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        """
        初始化Eagle推理模型
        
        Args:
            model_path: 模型路径，如果为None则使用项目内置模型
            device: 设备类型 ("auto", "cuda", "cpu")
            torch_dtype: 计算数据类型
            max_length: 生成文本最大长度
            temperature: 生成温度
            top_p: 核采样参数
            do_sample: 是否使用采样
        """
        self.model_path = model_path or self._get_default_model_path()
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        # 初始化模型和处理器
        self._load_model()
        self._setup_generation_config()
        
    def _get_default_model_path(self) -> str:
        """获取默认模型路径（项目内置的Eagle模型）"""
        project_root = Path(__file__).parent.parent
        return str(project_root / "gr00t" / "model" / "backbone" / "eagle2_hg_model")
    
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            print("警告：CUDA不可用，切换到CPU")
            device = "cpu"
            
        return torch.device(device)
    
    def _load_model(self):
        """加载模型和处理器"""
        print(f"正在加载Eagle模型从: {self.model_path}")
        
        try:
            # 加载配置
            self.config = Eagle2_5_VLConfig.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            
            # 加载处理器
            self.processor = Eagle2_5_VLProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            
            # 加载模型
            self.model = Eagle2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                config=self.config,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                device_map="auto" if self.device.type == "cuda" else None,
            )
            
            # 移动到指定设备
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print(f"模型加载成功，设备: {self.device}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def _setup_generation_config(self):
        """设置生成配置"""
        self.generation_config = GenerationConfig(
            max_length=self.max_length,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
    
    def _prepare_image(self, image_path: Union[str, Path, Image.Image]) -> Image.Image:
        """准备图像输入"""
        if isinstance(image_path, (str, Path)):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            image = image_path.convert("RGB")
        else:
            raise ValueError("不支持的图像输入类型")
        
        return image
    
    def _prepare_conversation(
        self, 
        text: str, 
        image: Optional[Image.Image] = None,
        system_prompt: Optional[str] = None
    ) -> List[dict]:
        """准备对话格式"""
        messages = []
        
        # 添加系统提示
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 准备用户消息
        user_content = []
        
        # 如果有图像，添加图像
        if image is not None:
            user_content.append({
                "type": "image",
                "image": image
            })
        
        # 添加文本
        if text.strip():
            user_content.append({
                "type": "text",
                "text": text
            })
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def generate_response(
        self,
        text: str,
        image: Optional[Union[str, Path, Image.Image]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        生成回复
        
        Args:
            text: 输入文本
            image: 输入图像（路径、PIL图像或None）
            system_prompt: 系统提示词
            max_new_tokens: 最大新生成token数
            **kwargs: 其他生成参数
            
        Returns:
            生成的回复文本
        """
        # 准备图像
        pil_image = None
        if image is not None:
            pil_image = self._prepare_image(image)
        
        # 准备对话格式
        messages = self._prepare_conversation(text, pil_image, system_prompt)
        
        # 应用聊天模板
        prompt = self.processor.py_apply_chat_template(
            messages, 
            add_generation_prompt=True
        )
        
        # 处理输入
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 更新生成配置
        generation_config = self.generation_config.copy()
        if max_new_tokens is not None:
            generation_config.max_new_tokens = max_new_tokens
        
        # 更新其他参数
        for key, value in kwargs.items():
            if hasattr(generation_config, key):
                setattr(generation_config, key, value)
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # 解码输出
        response = self.processor.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        # 提取生成的回复部分
        response = response.replace(prompt, "").strip()
        
        return response
    
    def image_description(
        self, 
        image: Union[str, Path, Image.Image],
        prompt: str = "请描述这张图片"
    ) -> str:
        """生成图像描述"""
        return self.generate_response(prompt, image)
    
    def visual_qa(
        self,
        question: str,
        image: Union[str, Path, Image.Image]
    ) -> str:
        """视觉问答"""
        return self.generate_response(question, image)
    
    def batch_process(
        self,
        tasks: List[dict]
    ) -> List[str]:
        """
        批量处理任务
        
        Args:
            tasks: 任务列表，每个任务包含text、image等字段
            
        Returns:
            回复列表
        """
        results = []
        
        for i, task in enumerate(tasks):
            print(f"处理任务 {i+1}/{len(tasks)}")
            
            try:
                response = self.generate_response(**task)
                results.append(response)
            except Exception as e:
                print(f"任务 {i+1} 处理失败: {e}")
                results.append(f"错误: {e}")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Eagle 2.5 视觉语言模型推理")
    
    # 模型参数
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=None,
        help="模型路径，默认使用项目内置模型"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="计算设备"
    )
    parser.add_argument(
        "--torch-dtype", 
        type=str, 
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="计算数据类型"
    )
    
    # 生成参数
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=512,
        help="生成文本最大长度"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="生成温度"
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=0.9,
        help="核采样参数"
    )
    parser.add_argument(
        "--max-new-tokens", 
        type=int, 
        default=None,
        help="最大新生成token数"
    )
    
    # 输入参数
    parser.add_argument(
        "--text", 
        type=str, 
        required=True,
        help="输入文本"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        default=None,
        help="输入图像路径"
    )
    parser.add_argument(
        "--system-prompt", 
        type=str, 
        default="你是一个有用的AI助手，能够理解和分析图像内容。",
        help="系统提示词"
    )
    
    # 输出参数
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="输出文件路径"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 设置数据类型
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype = dtype_map[args.torch_dtype]
    
    try:
        # 初始化模型
        print("正在初始化Eagle推理模型...")
        model = EagleVLInference(
            model_path=args.model_path,
            device=args.device,
            torch_dtype=torch_dtype,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        # 生成回复
        print("正在生成回复...")
        start_time = time.time()
        
        response = model.generate_response(
            text=args.text,
            image=args.image,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
        )
        
        end_time = time.time()
        
        # 输出结果
        print("\n" + "="*50)
        print("输入文本:", args.text)
        if args.image:
            print("输入图像:", args.image)
        print("系统提示:", args.system_prompt)
        print("-"*50)
        print("生成回复:")
        print(response)
        print("-"*50)
        print(f"生成耗时: {end_time - start_time:.2f}秒")
        print("="*50)
        
        # 保存到文件
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"输入文本: {args.text}\n")
                if args.image:
                    f.write(f"输入图像: {args.image}\n")
                f.write(f"系统提示: {args.system_prompt}\n")
                f.write(f"生成回复: {response}\n")
                f.write(f"生成耗时: {end_time - start_time:.2f}秒\n")
            print(f"结果已保存到: {args.output}")
        
        # 详细输出
        if args.verbose:
            print(f"\n模型配置:")
            print(f"  设备: {model.device}")
            print(f"  数据类型: {model.torch_dtype}")
            print(f"  最大长度: {model.max_length}")
            print(f"  温度: {model.temperature}")
            print(f"  Top-p: {model.top_p}")
            
    except Exception as e:
        print(f"错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

