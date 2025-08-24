#!/usr/bin/env python3
"""
修复版 Eagle 2.5 模型测试脚本

这个脚本使用更直接的方法来测试Eagle 2.5模型，避免复杂的对话模板处理。
"""

import os
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor

def test_eagle2_5_model_simple():
    """使用简单方法测试Eagle 2.5模型"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型路径
    model_path = "gr00t/model/backbone/eagle2_hg_model"
    
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        print("请确保在GR00T项目根目录下运行此脚本")
        return
    
    print(f"加载Eagle 2.5模型从: {model_path}")
    
    try:
        # 加载模型配置和处理器
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        
        # 设置tokenizer的padding方向
        processor.tokenizer.padding_side = "left"
        
        # 加载模型
        model = AutoModel.from_config(config, trust_remote_code=True)
        model.to(device)
        model.eval()
        
        print("模型加载完成!")
        
        # 测试图片路径
        test_image_path = "media/g1-pick-apple-images.png"
        
        if not os.path.exists(test_image_path):
            print(f"测试图片不存在: {test_image_path}")
            print("请提供一个有效的图片路径进行测试")
            return
        
        print(f"测试图片: {test_image_path}")
        
        # 加载图片
        image = Image.open(test_image_path).convert('RGB')
        
        # 方法1: 直接使用处理器处理图像和文本
        print("\n方法1: 直接处理图像和文本")
        
        # 处理图像
        image_inputs = processor(
            images=image,
            return_tensors="pt"
        )
        
        # 处理文本 - 使用简单的提示词
        text = "请描述这张图片的内容"
        text_inputs = processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # 移动到设备
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        pixel_values = image_inputs.pixel_values.to(device)
        
        print(f"输入形状: input_ids={input_ids.shape}, pixel_values={pixel_values.shape}")
        
        # 生成文本
        print("开始生成文字描述...")
        
        with torch.no_grad():
            outputs = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        # 解码生成的文本
        generated_text = processor.tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        print("\n" + "="*50)
        print("生成结果:")
        print(f"原始图片: {test_image_path}")
        print(f"提示词: {text}")
        print(f"生成的文字描述: {generated_text.strip()}")
        print("="*50)
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

def test_eagle2_5_model_with_conversation():
    """使用对话格式测试Eagle 2.5模型"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型路径
    model_path = "gr00t/model/backbone/eagle2_hg_model"
    
    try:
        # 加载模型
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        model = AutoModel.from_config(config, trust_remote_code=True)
        model.to(device)
        model.eval()
        
        print("模型加载完成!")
        
        # 测试图片路径
        test_image_path = "media/g1-pick-apple-images.png"
        
        if not os.path.exists(test_image_path):
            print(f"测试图片不存在: {test_image_path}")
            return
        
        print(f"测试图片: {test_image_path}")
        
        # 加载图片
        image = Image.open(test_image_path).convert('RGB')
        
        print("\n方法2: 使用对话格式")
        
        # 构建对话格式 - 使用图片路径而不是PIL对象
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": test_image_path},
                    {"type": "text", "text": "请详细描述这张图片的内容"}
                ]
            }
        ]
        
        # 应用聊天模板
        text = processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        print(f"处理后的文本: {text[:200]}...")
        
        # 分别处理文本和图像
        text_inputs = processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        image_inputs = processor(
            images=image,
            return_tensors="pt"
        )
        
        # 移动到设备
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        pixel_values = image_inputs.pixel_values.to(device)
        
        print(f"输入形状: input_ids={input_ids.shape}, pixel_values={pixel_values.shape}")
        
        # 生成文本
        print("开始生成文字描述...")
        
        with torch.no_grad():
            outputs = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        # 解码生成的文本
        generated_text = processor.tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        print("\n" + "="*50)
        print("对话格式生成结果:")
        print(f"原始图片: {test_image_path}")
        print(f"生成的文字描述: {generated_text.strip()}")
        print("="*50)
        
    except Exception as e:
        print(f"对话格式测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("修复版 Eagle 2.5 模型测试脚本")
    print("="*50)
    
    # 测试方法1: 直接处理
    test_eagle2_5_model_simple()
    
    print("\n" + "="*50)
    
    # 测试方法2: 对话格式
    test_eagle2_5_model_with_conversation()
