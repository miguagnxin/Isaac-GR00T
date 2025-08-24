#!/usr/bin/env python3
"""
简化版 Eagle 2.5 模型测试脚本

这个脚本用于测试Eagle 2.5模型在给定图片上生成文字描述的能力。
"""

import os
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor

def test_eagle2_5_model():
    """测试Eagle 2.5模型"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型路径 - 使用GR00T项目中的Eagle模型
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
        
        # 测试图片路径 - 使用项目中的示例图片
        test_image_path = "media/g1-pick-apple-images.png"
        
        if not os.path.exists(test_image_path):
            print(f"测试图片不存在: {test_image_path}")
            print("请提供一个有效的图片路径进行测试")
            return
        
        print(f"测试图片: {test_image_path}")
        
        # 加载和预处理图片
        image = Image.open(test_image_path).convert('RGB')
        
        # 构建对话格式
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image}, 
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
        
        # 处理输入
        inputs = processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # 移动到设备
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        # 处理视觉信息
        image_inputs, _ = processor.process_vision_info(conversation)
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
        print(f"生成的文字描述: {generated_text.strip()}")
        print("="*50)
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

def test_with_custom_image(image_path):
    """使用自定义图片测试模型"""
    
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在: {image_path}")
        return
    
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
        
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        
        # 构建对话
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": "请详细描述这张图片的内容"}
                ]
            }
        ]
        
        # 处理输入
        text = processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        image_inputs, _ = processor.process_vision_info(conversation)
        pixel_values = image_inputs.pixel_values.to(device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        # 解码结果
        generated_text = processor.tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        print(f"\n图片: {image_path}")
        print(f"描述: {generated_text.strip()}")
        
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    print("Eagle 2.5 模型测试脚本")
    print("="*50)
    
    # 测试默认图片
    test_eagle2_5_model()
    
    # 如果有命令行参数，测试指定的图片
    import sys
    if len(sys.argv) > 1:
        custom_image = sys.argv[1]
        print(f"\n测试自定义图片: {custom_image}")
        test_with_custom_image(custom_image)
