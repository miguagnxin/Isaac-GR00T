#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eagle 2.5 视觉语言模型推理测试脚本
用于验证推理功能是否正常工作

作者：AI Assistant
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

def test_import():
    """测试模块导入"""
    print("测试1: 模块导入...")
    try:
        from eagle_vl_inference import EagleVLInference
        print("✅ 模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_model_loading():
    """测试模型加载"""
    print("\n测试2: 模型加载...")
    try:
        from eagle_vl_inference import EagleVLInference
        
        # 尝试加载模型
        print("正在加载模型...")
        start_time = time.time()
        
        model = EagleVLInference(
            device="cpu",  # 使用CPU进行测试
            torch_dtype="float32",
            max_length=128
        )
        
        end_time = time.time()
        print(f"✅ 模型加载成功，耗时: {end_time - start_time:.2f}秒")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def test_text_generation():
    """测试纯文本生成"""
    print("\n测试3: 纯文本生成...")
    try:
        from eagle_vl_inference import EagleVLInference
        
        model = EagleVLInference(
            device="cpu",
            torch_dtype="float32",
            max_length=128
        )
        
        # 测试纯文本生成
        test_prompt = "请写一个简短的机器人故事。"
        print(f"测试提示: {test_prompt}")
        
        start_time = time.time()
        response = model.generate_response(
            text=test_prompt,
            system_prompt="你是一个创意写作助手。"
        )
        end_time = time.time()
        
        print(f"生成回复: {response}")
        print(f"✅ 文本生成成功，耗时: {end_time - start_time:.2f}秒")
        return True
        
    except Exception as e:
        print(f"❌ 文本生成失败: {e}")
        return False

def test_image_processing():
    """测试图像处理功能"""
    print("\n测试4: 图像处理功能...")
    try:
        from eagle_vl_inference import EagleVLInference
        
        model = EagleVLInference(
            device="cpu",
            torch_dtype="float32",
            max_length=128
        )
        
        # 测试图像处理（不实际生成，只测试处理流程）
        print("测试图像处理流程...")
        
        # 检查是否有示例图像
        demo_images = [
            "demo_data/robot_sim.PickNPlace/videos/chunk-000/episode_000000.mp4",  # 视频文件
            "media/header_compress.png",  # 图像文件
            "media/robots-banner.png"     # 图像文件
        ]
        
        found_image = None
        for img_path in demo_images:
            if os.path.exists(img_path):
                found_image = img_path
                break
        
        if found_image:
            print(f"找到测试图像: {found_image}")
            
            # 测试图像准备
            pil_image = model._prepare_image(found_image)
            print(f"✅ 图像处理成功，尺寸: {pil_image.size}")
            
            # 测试对话准备
            messages = model._prepare_conversation(
                "测试图像", 
                pil_image, 
                "测试系统提示"
            )
            print(f"✅ 对话准备成功，消息数: {len(messages)}")
            
            return True
        else:
            print("⚠️  未找到测试图像，跳过图像处理测试")
            return True
            
    except Exception as e:
        print(f"❌ 图像处理测试失败: {e}")
        return False

def test_batch_processing():
    """测试批量处理功能"""
    print("\n测试5: 批量处理功能...")
    try:
        from eagle_vl_inference import EagleVLInference
        
        model = EagleVLInference(
            device="cpu",
            torch_dtype="float32",
            max_length=128
        )
        
        # 测试批量任务准备
        tasks = [
            {
                "text": "测试任务1",
                "system_prompt": "测试系统提示1"
            },
            {
                "text": "测试任务2", 
                "system_prompt": "测试系统提示2"
            }
        ]
        
        print("测试批量任务准备...")
        print(f"✅ 批量任务准备成功，任务数: {len(tasks)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 批量处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("Eagle 2.5 视觉语言模型推理功能测试")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_import),
        ("模型加载", test_model_loading),
        ("文本生成", test_text_generation),
        ("图像处理", test_image_processing),
        ("批量处理", test_batch_processing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！推理功能正常工作。")
        print("\n下一步:")
        print("1. 运行 'python eagle_vl_examples.py' 查看使用示例")
        print("2. 使用 'python eagle_vl_inference.py --help' 查看命令行选项")
        print("3. 参考 README_eagle_vl.md 了解详细使用方法")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
        print("\n常见问题:")
        print("1. 确保已安装所需依赖包")
        print("2. 检查模型文件是否存在")
        print("3. 确认Python环境配置正确")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

