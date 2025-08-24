#!/usr/bin/env python3
"""
Eagle 2.5 模型文字生成测试脚本

这个脚本专门用于测试Eagle 2.5模型在给定测试集图片上生成文字内容的能力。
支持单张图片和多张图片的批量测试。

使用方法:
    python test_eagle2_5_text_generation.py --image_path path/to/image.jpg
    python test_eagle2_5_text_generation.py --image_dir path/to/image/directory
    python test_eagle2_5_text_generation.py --image_path path/to/image.jpg --prompt "请描述这张图片"
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
from PIL import Image
import numpy as np
from transformers import AutoConfig, AutoModel, AutoProcessor, GenerationConfig

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class Eagle2_5TextGenerator:
    """Eagle 2.5 模型文字生成器"""
    
    def __init__(self, model_path: str = None, max_length: int = 512):
        """
        初始化Eagle 2.5模型
        
        Args:
            model_path: 模型路径，如果为None则使用默认路径
            max_length: 生成文本的最大长度
        """
        if model_path is None:
            # 使用GR00T项目中的默认Eagle模型路径
            current_dir = Path(__file__).parent
            model_path = current_dir / "gr00t" / "model" / "backbone" / "eagle2_hg_model"
            
        self.model_path = Path(model_path)
        self.max_length = max_length
        
        print(f"加载Eagle 2.5模型从: {self.model_path}")
        
        # 加载模型配置和处理器
        self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, use_fast=True)
        
        # 设置tokenizer的padding方向
        self.processor.tokenizer.padding_side = "left"
        
        # 加载模型
        self.model = AutoModel.from_config(self.config, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        
        print("模型加载完成!")
        
        # 设置生成配置
        self.generation_config = GenerationConfig(
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            预处理后的图像tensor
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 使用处理器处理图像
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        
        return inputs.pixel_values.to(device)
    
    def generate_text(self, 
                     image_path: str, 
                     prompt: str = "请描述这张图片的内容",
                     max_new_tokens: int = 200) -> str:
        """
        基于图像生成文字描述
        
        Args:
            image_path: 图像文件路径
            prompt: 文本提示
            max_new_tokens: 最大生成的新token数量
            
        Returns:
            生成的文字描述
        """
        try:
            # 预处理图像
            pixel_values = self.preprocess_image(image_path)
            
            # 构建对话格式
            conversation = [
                {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}
            ]
            
            # 应用聊天模板
            text = self.processor.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 处理输入
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # 移动到设备
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            # 处理视觉信息
            image_inputs, _ = self.processor.process_vision_info(conversation)
            pixel_values = image_inputs.pixel_values.to(device)
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # 解码生成的文本
            generated_text = self.processor.tokenizer.decode(
                outputs[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"生成文字时出错: {e}")
            return f"错误: {str(e)}"
    
    def batch_generate(self, 
                       image_paths: List[str], 
                       prompt: str = "请描述这张图片的内容",
                       max_new_tokens: int = 200) -> List[Dict[str, Any]]:
        """
        批量生成文字描述
        
        Args:
            image_paths: 图像文件路径列表
            prompt: 文本提示
            max_new_tokens: 最大生成的新token数量
            
        Returns:
            包含生成结果的字典列表
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"处理第 {i+1}/{len(image_paths)} 张图片: {os.path.basename(image_path)}")
            
            start_time = time.time()
            generated_text = self.generate_text(image_path, prompt, max_new_tokens)
            end_time = time.time()
            
            result = {
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "prompt": prompt,
                "generated_text": generated_text,
                "generation_time": end_time - start_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            results.append(result)
            
            # 打印结果
            print(f"生成结果: {generated_text}")
            print(f"生成时间: {result['generation_time']:.2f}秒")
            print("-" * 50)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str = "eagle2_5_generation_results.json"):
        """
        保存生成结果到JSON文件
        
        Args:
            results: 生成结果列表
            output_file: 输出文件名
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {output_file}")


def get_image_files(image_dir: str) -> List[str]:
    """
    获取目录中的所有图像文件
    
    Args:
        image_dir: 图像目录路径
        
    Returns:
        图像文件路径列表
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    return [str(f) for f in sorted(image_files)]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试Eagle 2.5模型的文字生成能力")
    parser.add_argument("--image_path", type=str, help="单张图片的路径")
    parser.add_argument("--image_dir", type=str, help="包含多张图片的目录路径")
    parser.add_argument("--prompt", type=str, default="请详细描述这张图片的内容，包括场景、物体、动作等细节", 
                       help="生成文字的提示词")
    parser.add_argument("--max_new_tokens", type=int, default=200, 
                       help="最大生成的新token数量")
    parser.add_argument("--output_file", type=str, default="eagle2_5_generation_results.json",
                       help="输出结果文件名")
    
    args = parser.parse_args()
    
    # 检查输入参数
    if not args.image_path and not args.image_dir:
        print("错误: 必须指定 --image_path 或 --image_dir")
        parser.print_help()
        return
    
    # 初始化生成器
    try:
        generator = Eagle2_5TextGenerator()
    except Exception as e:
        print(f"初始化模型失败: {e}")
        return
    
    # 处理图片
    if args.image_path:
        # 单张图片
        if not os.path.exists(args.image_path):
            print(f"错误: 图片文件不存在: {args.image_path}")
            return
        
        print(f"处理单张图片: {args.image_path}")
        print(f"提示词: {args.prompt}")
        print("=" * 50)
        
        result = generator.generate_text(args.image_path, args.prompt, args.max_new_tokens)
        print(f"生成结果: {result}")
        
        # 保存结果
        results = [{
            "image_path": args.image_path,
            "image_name": os.path.basename(args.image_path),
            "prompt": args.prompt,
            "generated_text": result,
            "generation_time": 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }]
        
    else:
        # 多张图片
        if not os.path.exists(args.image_dir):
            print(f"错误: 目录不存在: {args.image_dir}")
            return
        
        image_files = get_image_files(args.image_dir)
        if not image_files:
            print(f"错误: 在目录 {args.image_dir} 中未找到图像文件")
            return
        
        print(f"找到 {len(image_files)} 张图片")
        print(f"提示词: {args.prompt}")
        print("=" * 50)
        
        # 批量生成
        results = generator.batch_generate(image_files, args.prompt, args.max_new_tokens)
    
    # 保存结果
    generator.save_results(results, args.output_file)
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("生成完成!")
    print(f"总共处理: {len(results)} 张图片")
    print(f"结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()
