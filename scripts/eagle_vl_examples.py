#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eagle 2.5 视觉语言模型使用示例
展示如何使用 eagle_vl_inference.py 脚本进行各种图文生成任务

作者：AI Assistant
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from eagle_vl_inference import EagleVLInference


def example_image_description():
    """示例：图像描述生成"""
    print("="*60)
    print("示例1: 图像描述生成")
    print("="*60)
    
    # 初始化模型
    model = EagleVLInference(
        device="auto",
        temperature=0.7,
        max_length=256
    )
    
    # 示例图像路径（需要用户提供真实图像）
    image_path = "path/to/your/image.jpg"  # 请替换为实际图像路径
    
    if os.path.exists(image_path):
        try:
            # 生成图像描述
            description = model.image_description(
                image=image_path,
                prompt="请详细描述这张图片的内容，包括场景、物体、颜色等细节。"
            )
            
            print(f"图像: {image_path}")
            print(f"描述: {description}")
            
        except Exception as e:
            print(f"图像描述生成失败: {e}")
    else:
        print(f"图像文件不存在: {image_path}")
        print("请提供有效的图像路径")


def example_visual_qa():
    """示例：视觉问答"""
    print("\n" + "="*60)
    print("示例2: 视觉问答")
    print("="*60)
    
    # 初始化模型
    model = EagleVLInference(
        device="auto",
        temperature=0.5,
        max_length=200
    )
    
    # 示例图像路径
    image_path = "path/to/your/image.jpg"  # 请替换为实际图像路径
    
    if os.path.exists(image_path):
        try:
            # 视觉问答
            questions = [
                "这张图片中有什么物体？",
                "图片中的主要颜色是什么？",
                "这张图片是在什么环境下拍摄的？",
                "图片中有什么文字或符号吗？"
            ]
            
            for question in questions:
                answer = model.visual_qa(question, image_path)
                print(f"问题: {question}")
                print(f"回答: {answer}")
                print("-" * 40)
                
        except Exception as e:
            print(f"视觉问答失败: {e}")
    else:
        print(f"图像文件不存在: {image_path}")
        print("请提供有效的图像路径")


def example_custom_conversation():
    """示例：自定义对话"""
    print("\n" + "="*60)
    print("示例3: 自定义对话")
    print("="*60)
    
    # 初始化模型
    model = EagleVLInference(
        device="auto",
        temperature=0.8,
        max_length=300
    )
    
    # 示例图像路径
    image_path = "path/to/your/image.jpg"  # 请替换为实际图像路径
    
    if os.path.exists(image_path):
        try:
            # 自定义系统提示
            system_prompt = "你是一个专业的图像分析师，擅长分析图像中的技术细节和艺术元素。"
            
            # 自定义对话
            conversations = [
                {
                    "text": "请分析这张图片的技术特点，包括构图、光线、色彩等方面。",
                    "image": image_path,
                    "system_prompt": system_prompt
                },
                {
                    "text": "如果这是一张产品图片，请给出改进建议。",
                    "image": image_path,
                    "system_prompt": system_prompt
                }
            ]
            
            for i, conv in enumerate(conversations, 1):
                print(f"对话 {i}:")
                response = model.generate_response(**conv)
                print(f"用户: {conv['text']}")
                print(f"助手: {response}")
                print("-" * 40)
                
        except Exception as e:
            print(f"自定义对话失败: {e}")
    else:
        print(f"图像文件不存在: {image_path}")
        print("请提供有效的图像路径")


def example_batch_processing():
    """示例：批量处理"""
    print("\n" + "="*60)
    print("示例4: 批量处理")
    print("="*60)
    
    # 初始化模型
    model = EagleVLInference(
        device="auto",
        temperature=0.6,
        max_length=200
    )
    
    # 示例图像路径
    image_path = "path/to/your/image.jpg"  # 请替换为实际图像路径
    
    if os.path.exists(image_path):
        try:
            # 批量任务
            tasks = [
                {
                    "text": "这张图片的主题是什么？",
                    "image": image_path,
                    "system_prompt": "你是一个图像主题分析专家。"
                },
                {
                    "text": "图片中有什么情感元素？",
                    "image": image_path,
                    "system_prompt": "你是一个情感分析专家。"
                },
                {
                    "text": "请用英文描述这张图片。",
                    "image": image_path,
                    "system_prompt": "You are an English image description expert."
                }
            ]
            
            print("开始批量处理...")
            results = model.batch_process(tasks)
            
            print("\n批量处理结果:")
            for i, (task, result) in enumerate(zip(tasks, results), 1):
                print(f"任务 {i}: {task['text']}")
                print(f"结果: {result}")
                print("-" * 40)
                
        except Exception as e:
            print(f"批量处理失败: {e}")
    else:
        print(f"图像文件不存在: {image_path}")
        print("请提供有效的图像路径")


def example_text_only_generation():
    """示例：纯文本生成（无图像）"""
    print("\n" + "="*60)
    print("示例5: 纯文本生成")
    print("="*60)
    
    # 初始化模型
    model = EagleVLInference(
        device="auto",
        temperature=0.9,
        max_length=400
    )
    
    try:
        # 纯文本对话
        system_prompt = "你是一个创意写作助手，擅长生成有趣的故事和描述。"
        
        text_prompts = [
            "请写一个关于机器人的短故事。",
            "描述一个未来城市的景象。",
            "写一首关于春天的诗。"
        ]
        
        for prompt in text_prompts:
            response = model.generate_response(
                text=prompt,
                system_prompt=system_prompt
            )
            print(f"提示: {prompt}")
            print(f"生成: {response}")
            print("-" * 40)
            
    except Exception as e:
        print(f"纯文本生成失败: {e}")


def main():
    """主函数"""
    print("Eagle 2.5 视觉语言模型使用示例")
    print("注意：请确保提供有效的图像文件路径")
    print()
    
    # 检查是否有图像文件
    demo_image = "path/to/your/image.jpg"  # 请替换为实际图像路径
    
    if not os.path.exists(demo_image):
        print("⚠️  警告：示例图像文件不存在")
        print(f"请将 {demo_image} 替换为实际的图像文件路径")
        print("或者修改脚本中的图像路径")
        print()
    
    # 运行示例
    try:
        # 纯文本生成示例（不需要图像）
        example_text_only_generation()
        
        # 需要图像的示例
        if os.path.exists(demo_image):
            example_image_description()
            example_visual_qa()
            example_custom_conversation()
            example_batch_processing()
        else:
            print("\n跳过需要图像的示例...")
            
    except Exception as e:
        print(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

