# PRC图像水印封装实现计划

## 🎯 项目目标
为统一多模态水印工具实现PRC图像水印封装，提供简洁易用的`embed`和`extract`接口。

## 📋 核心任务清单

### 阶段一：PRC水印核心封装 🔥 **高优先级**

#### ✅ 任务1：创建PRC水印核心封装类
**文件：** `src/image_watermark/prc_watermark.py`

**功能要求：**
- [ ] 实现`PRCWatermark`类的基础架构
- [ ] 实现`__init__(config)`初始化方法
- [ ] 实现`embed(prompt, message=None, **kwargs)`方法
  - [ ] 支持零位检测模式（`message=None`）
  - [ ] 支持长消息嵌入模式（字符串/二进制列表）
  - [ ] 自动消息格式转换（字符串→二进制）
  - [ ] 集成PRC密钥生成和管理
  - [ ] 集成Stable Diffusion图像生成
- [ ] 实现`extract(image)`方法
  - [ ] 图像逆向到潜在空间（exact_inversion）
  - [ ] 后验概率计算（recover_posteriors）
  - [ ] 水印检测（Detect函数）
  - [ ] 自动消息解码（Decode函数，检测到水印时）
  - [ ] 统一返回格式

**技术重点：**
- [ ] 自动密钥生成和缓存机制
- [ ] 参数自适应优化（基于消息长度）
- [ ] 完善的错误处理和状态报告
- [ ] 性能优化（模型加载、推理加速）

#### ✅ 任务2：辅助功能实现
**相关功能：**
- [ ] `_load_or_generate_keys()`：密钥管理
- [ ] `_message_to_binary()`：消息编码
- [ ] `_binary_to_message()`：消息解码  
- [ ] `_setup_diffusion_model()`：模型初始化


### 阶段二：系统集成 🔶 **中优先级**

#### ✅ 任务3：更新现有图像水印基类
**文件：** `src/image_watermark/image_watermark.py`

**修改内容：**
- [ ] 集成`PRCWatermark`算法支持
- [ ] 更新`embed_watermark()`方法实现
- [ ] 更新`extract_watermark()`方法实现
- [ ] 更新`generate_with_watermark()`方法
- [ ] 保持向后兼容性

#### ✅ 任务4：配置系统更新
**文件：** `config/default_config.yaml`

**新增配置项：**
- [ ] PRC算法专用配置段
- [ ] 模型路径和缓存设置
- [ ] 水印参数配置（误报率、稀疏度等）
- [ ] 性能优化参数

#### ✅ 任务5：统一引擎集成
**文件：** `src/unified/watermark_tool.py`

**集成内容：**
- [ ] 支持PRC图像水印算法选择
- [ ] 更新`generate_image_with_watermark()`方法
- [ ] 更新`extract_image_watermark()`方法
- [ ] 保持API一致性

### 阶段三：测试与优化 🔵 **一般优先级**

#### ✅ 任务6：示例和演示
**文件：** `examples/image_watermark_demo.py`

**演示内容：**
- [ ] 零位检测模式演示
- [ ] 长消息嵌入演示（字符串、二进制）
- [ ] 批量处理示例
- [ ] 性能基准测试
- [ ] 鲁棒性测试（JPEG压缩、高斯模糊等）

#### ✅ 任务7：单元测试
**文件：** `tests/test_prc_watermark.py`

**测试用例：**
- [ ] 密钥生成和加载测试
- [ ] 消息编码解码测试
- [ ] 图像生成测试
- [ ] 水印检测准确性测试
- [ ] 消息解码准确性测试
- [ ] 错误处理测试
- [ ] 性能基准测试

#### ✅ 任务8：文档和优化
- [ ] 更新`CLAUDE.md`中的PRC水印使用说明
- [ ] 性能分析和优化
- [ ] 内存使用优化
- [ ] GPU利用率优化

## 🔧 技术实现细节

### 关键依赖库
```python
# 核心依赖
torch>=1.10.0
diffusers>=0.10.0
transformers>=4.20.0
Pillow>=9.0.0

# PRC特定依赖
galois>=0.4.1
ldpc>=0.1.51
scipy>=1.7.0
```

### 配置示例
```yaml
image_watermark:
  algorithm: "prc"
  prc:
    model_name: "stabilityai/stable-diffusion-2-1-base"
    max_message_length: 512
    false_positive_rate: 1e-6
    sparsity: 3
    variance: 1.5
    num_inference_steps: 50
    guidance_scale: 7.5
    resolution: 512
    keys_dir: "./keys"
```

### 接口设计规范
```python
# 嵌入接口
result = watermark.embed(
    prompt="A beautiful sunset",
    message="Hello PRC!"  # 可选：None为零位模式
)
# 返回：{'watermarked_image': PIL.Image, 'success': bool, ...}

# 提取接口  
result = watermark.extract("image.png")
# 返回：{'detected': bool, 'decoded_message': str, 'confidence': float, ...}
```

## 📊 项目里程碑

- **里程碑1** (Week 1)：完成PRC水印核心封装类
- **里程碑2** (Week 2)：完成系统集成和配置更新
- **里程碑3** (Week 3)：完成测试、示例和文档

## 🚀 成功标准

1. **功能完整性**：支持零位检测和长消息嵌入两种模式
2. **接口一致性**：与现有文本水印接口风格保持一致
3. **性能表现**：图像生成<30秒，水印检测<10秒
4. **鲁棒性**：在常见攻击下保持>90%的检测率
5. **易用性**：提供清晰的文档和示例

## ⚠️ 风险和挑战

1. **模型加载时间**：Stable Diffusion模型较大，需要优化加载
2. **内存占用**：需要合理管理GPU内存使用
3. **参数调优**：不同消息长度需要不同的最优参数
4. **错误处理**：网络、模型、解码等多层次错误处理

## 📝 开发注意事项

- 遵循现有代码风格和命名规范
- 保持与文本水印接口的一致性
- 重视错误处理和用户体验
- 注重性能优化和资源管理
- 提供详细的日志和调试信息