# Eagle 2.5 视觉语言模型推理脚本使用说明

本目录包含了用于Eagle 2.5视觉语言模型推理的脚本，支持图文生成、视觉问答等多种任务。

## 文件说明

### 1. `eagle_vl_inference.py` - 主要推理脚本
- **功能**: Eagle 2.5模型的完整推理接口
- **特性**: 支持图像描述、视觉问答、自定义对话、批量处理
- **使用**: 命令行工具，支持多种参数配置

### 2. `eagle_vl_examples.py` - 使用示例脚本
- **功能**: 展示各种使用场景的示例代码
- **特性**: 包含5个主要示例，可直接运行学习
- **使用**: Python脚本，适合学习和开发参考

## 安装依赖

确保已安装以下依赖包：

```bash
pip install torch transformers pillow
```

## 使用方法

### 1. 命令行使用

#### 基本用法
```bash
# 纯文本生成
python eagle_vl_inference.py --text "请写一个关于机器人的故事"

# 图像描述
python eagle_vl_inference.py --text "请描述这张图片" --image "path/to/image.jpg"

# 视觉问答
python eagle_vl_inference.py --text "图片中有什么物体？" --image "path/to/image.jpg"
```

#### 高级参数
```bash
python eagle_vl_inference.py \
    --text "请分析这张图片" \
    --image "path/to/image.jpg" \
    --device cuda \
    --torch-dtype bfloat16 \
    --temperature 0.8 \
    --max-length 512 \
    --max-new-tokens 200 \
    --system-prompt "你是一个专业的图像分析师" \
    --output "result.txt" \
    --verbose
```

#### 参数说明
- `--text`: 输入文本（必需）
- `--image`: 输入图像路径（可选）
- `--model-path`: 模型路径，默认使用项目内置模型
- `--device`: 计算设备 (auto/cuda/cpu)
- `--torch-dtype`: 数据类型 (float32/float16/bfloat16)
- `--max-length`: 生成文本最大长度
- `--temperature`: 生成温度 (0.1-1.0)
- `--top-p`: 核采样参数 (0.1-1.0)
- `--max-new-tokens`: 最大新生成token数
- `--system-prompt`: 系统提示词
- `--output`: 输出文件路径
- `--verbose`: 详细输出

### 2. Python代码使用

#### 基本初始化
```python
from eagle_vl_inference import EagleVLInference

# 初始化模型
model = EagleVLInference(
    device="auto",           # 自动选择设备
    temperature=0.7,         # 生成温度
    max_length=512,          # 最大长度
    torch_dtype="bfloat16"   # 数据类型
)
```

#### 图像描述生成
```python
# 生成图像描述
description = model.image_description(
    image="path/to/image.jpg",
    prompt="请详细描述这张图片"
)
print(description)
```

#### 视觉问答
```python
# 视觉问答
answer = model.visual_qa(
    question="图片中有什么物体？",
    image="path/to/image.jpg"
)
print(answer)
```

#### 自定义对话
```python
# 自定义对话
response = model.generate_response(
    text="请分析这张图片的技术特点",
    image="path/to/image.jpg",
    system_prompt="你是一个专业的图像分析师",
    max_new_tokens=200
)
print(response)
```

#### 批量处理
```python
# 批量任务
tasks = [
    {
        "text": "图片主题是什么？",
        "image": "path/to/image.jpg",
        "system_prompt": "你是图像主题分析专家"
    },
    {
        "text": "有什么情感元素？",
        "image": "path/to/image.jpg",
        "system_prompt": "你是情感分析专家"
    }
]

results = model.batch_process(tasks)
for i, result in enumerate(results):
    print(f"任务 {i+1}: {result}")
```

### 3. 运行示例

```bash
# 运行所有示例
python eagle_vl_examples.py

# 注意：示例中的图像路径需要替换为实际路径
```

## 模型特性

### 支持的输入格式
- **图像**: JPG, PNG, BMP等常见格式
- **文本**: 中文、英文等多语言支持
- **对话**: 支持多轮对话和系统提示

### 生成能力
- **图像理解**: 场景识别、物体检测、情感分析
- **文本生成**: 故事创作、描述生成、问答回复
- **多模态**: 图文结合的理解和生成

### 性能优化
- **设备支持**: 自动GPU/CPU选择
- **数据类型**: 支持FP16/BF16优化
- **批处理**: 支持批量任务处理

## 注意事项

### 1. 硬件要求
- **GPU**: 推荐使用支持CUDA的GPU
- **内存**: 建议8GB以上显存
- **存储**: 模型文件约需2-3GB空间

### 2. 图像要求
- **格式**: 支持常见图像格式
- **大小**: 建议不超过4K分辨率
- **质量**: 清晰度越高效果越好

### 3. 常见问题
- **CUDA内存不足**: 尝试使用`--torch-dtype float16`
- **模型加载失败**: 检查模型路径和依赖安装
- **图像处理错误**: 确认图像文件存在且格式正确

## 扩展开发

### 1. 自定义模型
```python
# 使用自定义模型路径
model = EagleVLInference(
    model_path="/path/to/custom/model",
    device="cuda"
)
```

### 2. 自定义处理器
```python
# 继承并扩展推理类
class CustomEagleInference(EagleVLInference):
    def custom_method(self):
        # 自定义功能
        pass
```

### 3. 集成到其他系统
```python
# 作为模块导入
from eagle_vl_inference import EagleVLInference

# 在Web服务中使用
def web_api_handler(image, text):
    model = EagleVLInference()
    return model.generate_response(text, image)
```

## 技术支持

如果遇到问题，请检查：
1. 依赖包版本兼容性
2. 模型文件完整性
3. 硬件配置要求
4. 输入数据格式

## 更新日志

- **v1.0**: 初始版本，支持基本图文生成功能
- 支持多种输入格式和输出选项
- 提供完整的命令行接口和Python API
- 包含详细的使用示例和文档

