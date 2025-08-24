#!/usr/bin/env python3
"""
快速测试脚本 - 验证Eagle 2.5模型基本功能

这个脚本用于快速验证Eagle模型是否能正常加载和运行。
"""

import os
import sys
import torch

def check_dependencies():
    """检查必要的依赖包"""
    print("检查依赖包...")
    
    try:
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
    except ImportError:
        print("✗ transformers 未安装")
        return False
    
    try:
        import PIL
        print(f"✓ PIL: {PIL.__version__}")
    except ImportError:
        print("✗ PIL 未安装")
        return False
    
    try:
        print(f"✓ torch: {torch.__version__}")
    except:
        print("✗ torch 未安装")
        return False
    
    return True

def check_model_files():
    """检查模型文件是否存在"""
    print("\n检查模型文件...")
    
    model_path = "../gr00t/model/backbone/eagle2_hg_model"
    
    if not os.path.exists(model_path):
        print(f"✗ 模型路径不存在: {model_path}")
        return False
    
    required_files = [
        "config.json",
        "modeling_eagle2_5_vl.py",
        "processing_eagle2_5_vl.py",
        "configuration_eagle2_5_vl.py"
    ]
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} 缺失")
            return False
    
    return True

def test_model_loading():
    """测试模型加载"""
    print("\n测试模型加载...")
    
    try:
        from transformers import AutoConfig, AutoProcessor
        
        model_path = "../gr00t/model/backbone/eagle2_hg_model"
        
        # 测试配置加载
        print("加载模型配置...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ 配置加载成功: {config.model_type}")
        
        # 测试处理器加载
        print("加载模型处理器...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        print(f"✓ 处理器加载成功")
        
        # 测试tokenizer
        print("测试tokenizer...")
        test_text = "Hello, world!"
        tokens = processor.tokenizer(test_text)
        print(f"✓ Tokenizer测试成功: {len(tokens['input_ids'])} tokens")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False

def test_device():
    """测试设备可用性"""
    print("\n检查设备...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ CUDA不可用，将使用CPU")
    
    print(f"✓ 当前设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    return True

def main():
    """主函数"""
    print("Eagle 2.5 模型快速测试")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请安装必要的包")
        print("运行: pip install torch transformers pillow")
        return False
    
    # 检查模型文件
    if not check_model_files():
        print("\n❌ 模型文件检查失败")
        print("请确保在GR00T项目根目录下运行此脚本")
        return False
    
    # 检查设备
    test_device()
    
    # 测试模型加载
    if not test_model_loading():
        print("\n❌ 模型加载测试失败")
        return False
    
    print("\n" + "=" * 50)
    print("✅ 所有测试通过！Eagle 2.5模型可以正常使用")
    print("\n现在可以运行以下命令进行完整测试:")
    print("python simple_eagle2_5_test.py")
    print("python test_eagle2_5_text_generation.py --image_path path/to/image.jpg")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
