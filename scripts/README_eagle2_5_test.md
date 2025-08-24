# Eagle 2.5 模型文字生成测试

这个目录包含了用于测试GR00T项目中Eagle 2.5模型文字生成能力的脚本。

## 脚本说明

### 1. `simple_eagle2_5_test.py` - 简化版测试脚本
- **功能**: 测试Eagle 2.5模型在给定图片上生成文字描述
- **特点**: 简单易用，适合快速测试
- **默认测试**: 使用项目中的示例图片 `media/g1-pick-apple-images.png`

### 2. `test_eagle2_5_text_generation.py` - 完整版测试脚本
- **功能**: 支持单张图片和批量图片的文字生成测试
- **特点**: 功能完整，支持自定义提示词和批量处理
- **输出**: 生成JSON格式的详细结果报告

## 使用方法

### 环境要求
```bash
# 确保已安装必要的依赖
pip install torch transformers pillow numpy
```

### 简化版测试
```bash
# 测试默认图片
python simple_eagle2_5_test.py

# 测试自定义图片
python simple_eagle2_5_test.py path/to/your/image.jpg
```

### 完整版测试
```bash
# 测试单张图片
python test_eagle2_5_text_generation.py --image_path path/to/image.jpg

# 测试目录中的所有图片
python test_eagle2_5_text_generation.py --image_dir path/to/image/directory

# 自定义提示词
python test_eagle2_5_text_generation.py --image_path path/to/image.jpg --prompt "请分析这张图片中的机器人动作"

# 自定义生成参数
python test_eagle2_5_text_generation.py --image_path path/to/image.jpg --max_new_tokens 300
```

## 参数说明

### 简化版脚本参数
- 无命令行参数：使用默认测试图片
- 第一个参数：自定义图片路径

### 完整版脚本参数
- `--image_path`: 单张图片的路径
- `--image_dir`: 包含多张图片的目录路径
- `--prompt`: 生成文字的提示词（默认：请详细描述这张图片的内容）
- `--max_new_tokens`: 最大生成的新token数量（默认：200）
- `--output_file`: 输出结果文件名（默认：eagle2_5_generation_results.json）

## 输出结果

### 简化版脚本
- 直接在控制台显示生成结果
- 包含图片路径和生成的文字描述

### 完整版脚本
- 生成JSON格式的结果文件
- 包含每张图片的详细信息：
  - 图片路径和名称
  - 使用的提示词
  - 生成的文字描述
  - 生成时间
  - 时间戳

## 示例输出

### 控制台输出
```
Eagle 2.5 模型测试脚本
==================================================
使用设备: cuda
加载Eagle 2.5模型从: gr00t/model/backbone/eagle2_hg_model
模型加载完成!
测试图片: media/g1-pick-apple-images.png
处理后的文本: <|im_start|>user<|im_end|><|im_start|>assistant<|im_end|>...
输入形状: input_ids=torch.Size([1, 67]), pixel_values=torch.Size([1, 3, 224, 224])
开始生成文字描述...

==================================================
生成结果:
原始图片: media/g1-pick-apple-images.png
生成的文字描述: 这张图片显示了一个机器人手臂正在执行抓取苹果的任务...
==================================================
```

### JSON结果文件
```json
[
  {
    "image_path": "media/g1-pick-apple-images.png",
    "image_name": "g1-pick-apple-images.png",
    "prompt": "请详细描述这张图片的内容，包括场景、物体、动作等细节",
    "generated_text": "这张图片显示了一个机器人手臂正在执行抓取苹果的任务...",
    "generation_time": 2.45,
    "timestamp": "2024-01-15 14:30:25"
  }
]
```

## 注意事项

1. **模型路径**: 确保在GR00T项目根目录下运行脚本
2. **图片格式**: 支持常见的图片格式（jpg, png, bmp, tiff, webp）
3. **内存要求**: Eagle 2.5模型较大，建议使用GPU运行
4. **首次运行**: 首次运行可能需要下载模型权重，请确保网络连接正常

## 故障排除

### 常见问题

1. **模型路径错误**
   ```
   错误: 模型路径不存在: gr00t/model/backbone/eagle2_hg_model
   ```
   - 解决方案：确保在GR00T项目根目录下运行脚本

2. **依赖缺失**
   ```
   ModuleNotFoundError: No module named 'transformers'
   ```
   - 解决方案：安装必要的依赖包

3. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   - 解决方案：减少batch size或使用CPU运行

4. **图片加载失败**
   ```
   PIL.UnidentifiedImageError: cannot identify image file
   ```
   - 解决方案：检查图片文件是否损坏或格式不支持

## 扩展功能

### 自定义提示词示例
```python
# 场景分析
prompt = "请分析这张图片中的机器人工作环境，包括安全性和效率性"

# 动作识别
prompt = "请识别并描述图片中机器人的具体动作步骤"

# 物体检测
prompt = "请列出图片中所有可见的物体和工具"

# 任务理解
prompt = "请解释这张图片中机器人正在执行什么任务"
```

### 批量处理优化
- 对于大量图片，建议分批处理以避免内存溢出
- 可以调整`max_new_tokens`参数控制生成文本的长度
- 使用`--output_file`参数保存结果以便后续分析
