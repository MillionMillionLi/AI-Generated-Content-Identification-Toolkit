好的！按照这个目录结构重新制定十天计划。这个结构更有条理，我来为你设计一个适合的开发计划。

好的！根据你选择的标准目录结构，我来重新制定一个更合理的十天开发计划。
Day 1: 项目搭建 - 创建极简目录结构，设置requirements.txt，初始化基本文件
Day 2: 复制CredID代码 - 从原项目复制核心文件，修复导入路径问题
Day 3: 复制Stable Signature代码 - 复制核心文件，下载预训练解码器
Day 4-5: 文本水印封装 - 创建TextWatermark类，硬编码配置，实现嵌入和提取接口
Day 6-7: 图像水印封装 - 创建ImageWatermark类，实现图像生成和水印提取
Day 8: 统一工具类 - 创建WatermarkTool类，整合文本和图像水印功能
Day 9: 编写demo - 创建unified_demo.py，text_demo.py，image_demo.py


## 📅 十天开发计划 - 标准版

### 🏗️ Day 1-2: 项目基础设施

#### Day 1: 目录结构和基础工具
**任务**:
```bash
# 1. 创建项目结构
mkdir mmwt && cd mmwt
mkdir -p {config,src/{text_watermark,image_watermark,utils},examples,tests,models}

# 2. 创建__init__.py文件
find src -type d -exec touch {}/__init__.py \;
touch __init__.py

# 3. 基础文件
touch {README.md,requirements.txt,setup.py}
touch config/{text_config.yaml,image_config.yaml}
touch src/watermark_engine.py
```

**requirements.txt**:
```
torch>=1.12.0
transformers>=4.19.2
diffusers>=0.20.0
numpy>=1.20.0
scipy>=1.7.0
Pillow>=8.0.0
pyyaml>=5.4.0
tqdm>=4.65.0
omegaconf>=2.1.1
scikit-image>=0.20.0
```

#### Day 2: 基础工具模块
实现 `src/utils/config_loader.py` 和 `src/utils/model_manager.py`

```python
# src/utils/config_loader.py
import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
```

---

### 📝 Day 3-5: CredID文本水印集成 ✅ **已完成**

#### Day 3: CredID代码复制和修复 ✅
**已完成任务:**
```bash
# 1. 复制CredID核心文件到 src/text_watermark/credid/
# 从 credible_LLM_watermarking 项目复制以下文件:
# - watermarking/CredID/ (完整目录)
# - src/utils/ (工具函数)

# 2. 修复导入路径问题
# 调整所有相对导入路径：from src.utils.* → from ...src.utils.*
```

**核心文件结构:**
```
src/text_watermark/credid/
├── __init__.py                     # 包初始化，暴露核心组件
├── watermarking/CredID/
│   ├── message_model_processor.py  # 核心处理器 (WmProcessorMessageModel)
│   ├── random_message_model_processor.py  # 随机处理器
│   ├── base_processor.py          # 基础接口
│   ├── random_processor.py        # 简单随机处理器
│   └── message_models/
│       ├── lm_message_model.py    # LM消息模型 (LMMessageModel)
│       ├── random_message_model.py # 随机消息模型
│       ├── base_message_model.py  # 基础消息模型
│       └── base_message_model_fast.py
└── src/utils/                     # CredID工具函数
    ├── hash_fn.py                 # 哈希函数
    └── random_utils.py            # 随机工具
```

#### Day 4: CredID封装类开发 ✅
**思考：嵌入水印长度的问题，过短会怎么样，为什么之前的测试只嵌入一个也能正常运行；如果多退少补，extract的时候怎么办**
**实现的完整接口:** `src/text_watermark/credid_watermark.py`

```python
class CredIDWatermark:
    """
    CredID文本水印算法统一封装
    
    功能特点:
    - 支持多种消息格式 (字符串、整数列表、字符串列表)
    - 支持两种模式: LM模式(高质量) / Random模式(高速度)
    - 智能多段消息处理和候选消息优化搜索
    - 完整的错误处理和置信度评估
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化CredID水印处理器
        
        Args:
            config: 配置字典，包含模型和算法参数
                - mode: 'lm' 或 'random' (默认'lm')
                - model_name: 预训练模型名称
                - lm_params: LM模式参数 (delta, prefix_len, message_len等)
                - wm_params: 水印处理参数 (encode_ratio, strategy等)
        """
    
    def embed(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
              prompt: str, message: Union[str, List[int], List[str]], 
              segmentation_mode: str = 'auto') -> Dict[str, Any]:
        """
        🔹 核心功能: 在文本生成过程中嵌入水印
        
        工作流程:
        1. 将消息转换为CredID兼容的二进制格式 (支持多段)
        2. 设置LogitsProcessorList，集成水印处理器
        3. 使用model.generate()生成带水印文本
        4. 返回完整结果和元数据
        
        Args:
            model: HuggingFace语言模型
            tokenizer: 对应的分词器  
            prompt: 输入提示文本
            message: 水印信息，支持:
                - str: "hello" 或 "log20250725143000"
                - List[int]: [123, 456, 789]
                - List[str]: ["user", "2025", "admin"]
            segmentation_mode: 消息分割模式
                - 'auto': 自动判断 (默认)
                - 'smart': 智能分割 (如 "alibaba20250725" → ["alibaba", "2025", "0725"])
                - 'whole': 整体处理
                - 'spaces': 按空格分割
                
        Returns:
            {
                'watermarked_text': str,      # 带水印的生成文本
                'original_message': Any,      # 原始水印信息
                'binary_message': List[int],  # 转换后的二进制消息
                'prompt': str,                # 输入提示
                'success': bool,              # 是否成功
                'metadata': {                 # 生成元数据
                    'mode': str,              # 使用的模式 ('lm'/'random')
                    'model_name': str,        # 模型名称
                    'input_length': int,      # 输入长度
                    'output_length': int,     # 输出长度
                    'generation_config': dict # 生成配置
                }
            }
        """
    
    def extract(self, watermarked_text: str, 
                model: Optional[PreTrainedModel] = None,
                tokenizer: Optional[PreTrainedTokenizer] = None,
                candidates_messages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        🔹 核心功能: 从水印文本中提取水印信息
        
        工作流程:
        1. 检查模式和参数有效性
        2. 收集候选消息的所有编码 (如果提供candidates_messages)
        3. 使用CredID解码器进行统计检测
        4. 智能匹配候选消息或重构原始消息
        5. 计算置信度并返回结果
        
        Args:
            watermarked_text: 可能包含水印的文本
            model: 语言模型 (LM模式必需，Random模式可选)
            tokenizer: 分词器 (LM模式必需，Random模式可选)
            candidates_messages: 候选消息列表，用于优化搜索
                例如: ["log20250725143000", "user987654321", "admin2025"]
                
        Returns:
            {
                'extracted_message': str,           # 提取的消息
                'binary_message': List[int],        # 解码的二进制消息
                'confidence': float,                # 置信度 (0.0-1.0)
                'success': bool,                    # 是否成功提取
                'detailed_confidence': List,       # 详细置信度信息
                'metadata': {
                    'mode': str,                    # 检测模式
                    'text_length': int,             # 文本长度
                    'num_decoded_segments': int,    # 解码段数
                    'detection_method': 'CredID',   # 检测方法
                    'confidence_threshold': float,  # 置信度阈值
                    'search_space': int,            # 搜索空间大小
                    'candidates_provided': bool     # 是否提供候选消息
                }
            }
        """
    
    # 内部方法
    def _message_to_binary(self, message, segmentation_mode='auto') -> List[int]
    def _binary_to_message(self, binary: List[int]) -> str
    def _match_decoded_with_candidates(self, decoded_messages, candidates) -> Tuple[str, float]
    def _calculate_sequence_match(self, decoded, candidate) -> float
```

#### Day 5: 配置系统和测试 ✅
**配置文件:** `config/text_config.yaml`
```yaml
# === 基础配置 ===
method: "CredID"                          # 水印算法
model_name: "huggyllama/llama-7b"         # 语言模型
mode: "lm"                                # 模式: 'lm'(高质量) / 'random'(高速度)
device: "auto"                            # 设备: 'auto'/'cuda'/'cpu'

# === 生成参数 ===
max_new_tokens: 110                       # 最大生成token数
num_beams: 4                              # beam search宽度
do_sample: true                           # 是否采样
temperature: 0.7                          # 生成温度
top_p: 0.9                               # nucleus采样
top_k: 50                                # top-k采样

# === CredID LM模式参数 ===
lm_params:
  delta: 1.5                              # logits修改强度
  prefix_len: 10                          # 前缀保护长度
  message_len: 10                         # 消息二进制长度 (位)
  seed: 42                                # 随机种子
  topk: -1                               # LM top-k限制
  permutation_num: 50                     # 随机排列数
  hash_prefix_len: 1                      # 哈希前缀长度
  shifts: [21, 24, 3, 8, 14, 2, 4, 28, 31, 3, 8, 14, 2, 4, 28]  # 哈希移位

# === 水印处理参数 ===
wm_params:
  encode_ratio: 8                         # 编码比率 (每消息位对应的token数)
  seed: 42                                # 水印种子
  strategy: "vanilla"                     # 策略: 'vanilla'/'max_confidence'
  max_confidence: 0.5                     # 最大置信度阈值
  top_k: 1000                            # 处理器top-k

# === 解码配置 ===
decode_batch_size: 16                     # 解码批次大小
disable_tqdm: false                       # 是否禁用进度条
confidence_threshold: 0.6                 # 成功检测的置信度阈值
```

**使用示例和测试:**
```python
# === 基础使用示例 ===
from src.text_watermark.credid_watermark import CredIDWatermark
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

# 1. 加载配置和模型
with open('config/text_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 初始化水印
watermark = CredIDWatermark(config)

# 3. 嵌入水印 - 支持多种消息格式
results = []

# 简单字符串
result1 = watermark.embed(model, tokenizer, "Hello, today is", "tech")
results.append(("简单消息", result1))

# 复杂字符串 (自动智能分割)
result2 = watermark.embed(model, tokenizer, "System log entry:", "log20250725143000")
results.append(("复杂消息", result2))

# 列表消息
result3 = watermark.embed(model, tokenizer, "User information:", ["admin", "2025", "secure"])
results.append(("列表消息", result3))

# 4. 提取水印
for desc, result in results:
    if result['success']:
        print(f"\n=== {desc} ===")
        print(f"生成文本: {result['watermarked_text']}")
        
        # 基础提取
        extracted = watermark.extract(result['watermarked_text'], model, tokenizer)
        print(f"提取结果: {extracted['extracted_message']}")
        print(f"置信度: {extracted['confidence']:.3f}")
        print(f"成功: {extracted['success']}")
        
        # 候选消息优化提取 (可选)
        candidates = ["tech", "log20250725143000", "admin2025secure", "hello", "test"]
        extracted_opt = watermark.extract(
            result['watermarked_text'], 
            model, tokenizer, 
            candidates_messages=candidates
        )
        print(f"优化提取: {extracted_opt['extracted_message']} (置信度: {extracted_opt['confidence']:.3f})")

# === 高级功能示例 ===
# 1. 批量处理
messages = ["hello", "tech2025", "user123admin", "log20250725143000"]
prompts = ["Today is", "System:", "User login:", "Entry log:"]

batch_results = []
for prompt, message in zip(prompts, messages):
    result = watermark.embed(model, tokenizer, prompt, message)
    if result['success']:
        extracted = watermark.extract(result['watermarked_text'], model, tokenizer)
        batch_results.append({
            'original': message,
            'extracted': extracted['extracted_message'],
            'confidence': extracted['confidence'],
            'match': message == extracted['extracted_message']
        })

print(f"\n=== 批量处理结果 ===")
for i, result in enumerate(batch_results):
    status = "✅" if result['match'] else "❌"
    print(f"{status} {result['original']} → {result['extracted']} (置信度: {result['confidence']:.3f})")

# 2. 性能测试
import time

print(f"\n=== 性能测试 ===")
start_time = time.time()
for i in range(10):
    result = watermark.embed(model, tokenizer, f"Test {i}:", f"msg{i}")
    if result['success']:
        watermark.extract(result['watermarked_text'], model, tokenizer)
end_time = time.time()

print(f"10次嵌入+提取耗时: {end_time - start_time:.2f}秒")
print(f"平均每次: {(end_time - start_time)/10:.2f}秒")
```

**🎯 Day 3-5 总结:**
- ✅ **代码集成**: 成功复制并修复CredID核心代码
- ✅ **接口封装**: 实现统一的embed/extract接口
- ✅ **多消息支持**: 支持字符串、列表、复杂混合消息
- ✅ **候选优化**: 实现候选消息限制搜索提升效率
- ✅ **错误处理**: 完整的异常处理和状态报告
- ✅ **配置系统**: 灵活的YAML配置管理
- ✅ **测试验证**: 多种场景的功能测试

**为后续开发提供的参考架构:**
1. **统一接口模式**: embed(model, tokenizer, prompt, message) → extract(text, model, tokenizer)
2. **配置驱动设计**: 通过YAML文件管理算法参数
3. **返回格式标准**: 统一的成功/失败状态和元数据结构
4. **候选优化机制**: 支持候选列表的高效搜索
5. **多模态消息**: 支持多种输入格式的消息编码

---

### 🖼️ Day 6-8: Stable Signature图像水印集成

#### Day 6: Stable Signature代码复制
```bash
# 复制核心文件
cp /path/to/stable_signature/utils*.py src/image_watermark/stable_sig/
cp /path/to/stable_signature/finetune_ldm_decoder.py src/image_watermark/stable_sig/
cp -r /path/to/stable_signature/src/* src/image_watermark/stable_sig/

# 下载预训练模型
wget https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt -P models/
```

#### Day 7: Stable Signature封装类开发
实现 `src/image_watermark/stable_signature.py`

```python
# src/image_watermark/stable_signature.py
import torch
from PIL import Image
from typing import Dict, Any, Union
from diffusers import StableDiffusionPipeline

class StableSignatureWatermark:
    """Stable Signature图像水印算法封装"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_models()
    
    def embed(self, prompt: str, message: str) -> Dict[str, Any]:
        """生成带水印图像"""
        # 实现图像生成和水印嵌入
        pass
    
    def extract(self, watermarked_image: Union[Image.Image, str]) -> Dict[str, Any]:
        """提取图像水印"""
        # 实现水印提取
        pass
```

#### Day 8: 图像水印测试
**config/image_config.yaml**:
```yaml
method: "stable_signature"
message_length: 48
decoder_path: "models/dec_48b_whit.torchscript.pt"

diffusion:
  model_name: "stabilityai/stable-diffusion-2"
  num_inference_steps: 50
  guidance_scale: 7.5
  height: 512
  width: 512
```

---

### 🚀 Day 9-10: 统一引擎和整合

#### Day 9: WatermarkEngine统一接口
实现 `src/watermark_engine.py`

```python
# src/watermark_engine.py
import os
from typing import Dict, Any
from .utils.config_loader import load_config
from .text_watermark.credid_watermark import CredIDWatermark
from .image_watermark.stable_signature import StableSignatureWatermark

class WatermarkEngine:
    """多模态水印统一引擎"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.text_watermark = None
        self.image_watermark = None
    
    def embed_text(self, model, tokenizer, prompt: str, message: str):
        """嵌入文本水印"""
        if not self.text_watermark:
            config = load_config(os.path.join(self.base_dir, "config/text_config.yaml"))
            self.text_watermark = CredIDWatermark(config)
        return self.text_watermark.embed(model, tokenizer, prompt, message)
    
    def extract_text(self, watermarked_text: str):
        """提取文本水印"""
        if not self.text_watermark:
            config = load_config(os.path.join(self.base_dir, "config/text_config.yaml"))
            self.text_watermark = CredIDWatermark(config)
        return self.text_watermark.extract(watermarked_text)
    
    def embed_image(self, prompt: str, message: str):
        """嵌入图像水印"""
        if not self.image_watermark:
            config = load_config(os.path.join(self.base_dir, "config/image_config.yaml"))
            self.image_watermark = StableSignatureWatermark(config)
        return self.image_watermark.embed(prompt, message)
    
    def extract_image(self, watermarked_image):
        """提取图像水印"""
        if not self.image_watermark:
            config = load_config(os.path.join(self.base_dir, "config/image_config.yaml"))
            self.image_watermark = StableSignatureWatermark(config)
        return self.image_watermark.extract(watermarked_image)
```

#### Day 10: 最终测试和文档
创建 `examples/unified_demo.py`

```python
# examples/unified_demo.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.watermark_engine import WatermarkEngine
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("🌊 多模态水印工具演示")
    
    # 初始化引擎
    engine = WatermarkEngine()
    
    # 文本水印演示
    print("\n📝 文本水印测试...")
    model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    
    result = engine.embed_text(model, tokenizer, "The weather today is", "hello")
    print(f"水印文本: {result}")
    
    extracted = engine.extract_text(result['watermarked_text'])
    print(f"提取结果: {extracted}")
    
    # 图像水印演示
    print("\n🖼️ 图像水印测试...")
    image_result = engine.embed_image("a beautiful cat", "secret")
    print(f"图像生成: {image_result}")
    
    extracted = engine.extract_image(image_result['watermarked_image'])
    print(f"提取结果: {extracted}")

if __name__ == "__main__":
    main()
```
Ran tool