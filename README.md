# 统一多模态水印工具 (Unified Watermark Tool)

一个支持文本和图像水印的统一工具库，集成了多种先进的水印算法。

## 功能特性

### 文本水印
- **CredID算法**: 基于可信身份的文本水印
- **批量处理**: 支持文本批量水印嵌入和提取
- **质量评估**: 提供水印质量和鲁棒性评估

### 图像水印
- **Stable Signature**: 基于扩散模型的图像水印
- **高质量生成**: 保持图像质量的同时嵌入水印
- **可靠检测**: 准确的水印提取和验证

### 统一接口
- **简单易用**: 统一的API接口设计
- **灵活配置**: 支持多种参数配置
- **扩展性强**: 易于添加新的水印算法

## 项目结构

```
unified_watermark_tool/
├── src/
│   ├── text_watermark/      # 文本水印模块
│   ├── image_watermark/     # 图像水印模块
│   └── unified/             # 统一接口模块
├── tests/                   # 测试代码
├── examples/                # 示例代码
├── docs/                    # 文档
├── config/                  # 配置文件
└── requirements.txt         # 依赖文件
```

## 快速开始

### 安装

```bash
git clone <repository_url>
cd unified_watermark_tool
pip install -r requirements.txt
```

### 使用示例

```python
from src import WatermarkTool

# 创建统一水印工具
tool = WatermarkTool()

# 文本水印
watermarked_text = tool.embed_text_watermark("原始文本", watermark_key="my_key")
detected_watermark = tool.extract_text_watermark(watermarked_text, watermark_key="my_key")

# 图像水印
watermarked_image = tool.embed_image_watermark("原始图像路径", watermark_key="my_key")
detected_watermark = tool.extract_image_watermark("水印图像路径", watermark_key="my_key")
```

## 开发计划

- [x] Day 1: 项目搭建
- [ ] Day 2: CredID代码集成
- [ ] Day 3: Stable Signature集成
- [ ] Day 4-5: 文本水印封装
- [ ] Day 6-7: 图像水印封装
- [ ] Day 8: 统一工具类
- [ ] Day 9: 示例代码
- [ ] Day 10: 测试和文档

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License 