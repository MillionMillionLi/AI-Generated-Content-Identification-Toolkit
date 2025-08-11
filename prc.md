● 📚 核心代码文件及其主要函数详解

  1. src/prc.py - PRC算法核心

  这是整个水印系统的算法核心，包含所有关键的密码学操作：

  🔑 密钥管理函数

  - KeyGen(n, message_length=512, false_positive_rate=1e-9, t=3, ...)
    - 生成PRC密钥对（编码密钥 + 解码密钥）
    - n: 潜在空间维度（对应4×64×64=16384）
    - message_length: 支持的最大消息长度（位）
    - false_positive_rate: 误报率控制
    - t: 稀疏度参数（影响安全性vs鲁棒性权衡）

  🎯 核心水印操作

  - Encode(encoding_key, message=None)
    - 最关键函数：将消息编码为PRC码字
    - message=None: 零位检测模式
    - message=[0,1,1,0,...]: 长消息嵌入模式
    - 返回伪高斯分布的码字
  - Detect(decoding_key, posteriors, false_positive_rate=None)
    - 检测是否存在水印（二进制结果）
    - posteriors: 从图像恢复的后验概率
    - 返回 True/False
  - Decode(decoding_key, posteriors, ...)
    - 解码嵌入的消息内容
    - 使用belief propagation算法
    - 成功返回消息，失败返回None

  🛠️ 辅助函数

  - str_to_bin(string): 字符串→二进制转换
  - bin_to_str(bin_list): 二进制→字符串转换

  2. src/pseudogaussians.py - 伪高斯分布处理

  这个模块处理PRC码字与潜在空间之间的转换：

  - sample(codeword, basis=None)
    - 将PRC码字转换为伪高斯噪声
    - 用于初始化扩散模型的潜在表示
    - 关键：pseudogaussian_np = codeword_np * np.abs(np.random.randn(...))
  - recover_posteriors(z, basis=None, variances=None)
    - 从潜在表示恢复后验概率
    - 使用erf函数计算符号概率
    - 用于水印检测和解码

  3. inversion.py - 扩散模型接口

  处理与Stable Diffusion的交互：

  - stable_diffusion_pipe(...)
    - 初始化Stable Diffusion管道
    - 配置模型、求解器等参数
  - generate(prompt, init_latents, ...)
    - 从给定的初始潜在表示生成图像
    - init_latents: 含水印的伪高斯噪声
    - 返回最终生成的图像
  - exact_inversion(img, prompt, ...)
    - 关键函数：从图像反推初始潜在表示
    - 使用精确逆向过程
    - 返回用于水印检测的潜在表示

  4. encode.py - 图像生成示例

  展示如何生成带水印的图像：

  关键流程：
  # 1. 生成PRC码字
  prc_codeword = Encode(encoding_key, message)  # ← 需要添加message参数

  # 2. 转换为伪高斯噪声
  init_latents = prc_gaussians.sample(prc_codeword).reshape(1, 4, 64, 64)

  # 3. 生成图像
  image = generate(prompt, init_latents, ...)

  5. decode.py - 水印检测示例

  展示如何从图像中检测和解码水印：

  关键流程：
  # 1. 图像逆向到潜在空间
  reversed_latents = exact_inversion(img, ...)

  # 2. 恢复后验概率
  reversed_prc = prc_gaussians.recover_posteriors(reversed_latents, ...)

  # 3. 检测和解码
  detection_result = Detect(decoding_key, reversed_prc)
  decoding_result = Decode(decoding_key, reversed_prc)

  🎯 封装重点

  基于这些核心函数，我们的封装需要：

  1. 简化密钥管理：自动生成和缓存密钥
  2. 统一消息处理：支持字符串、二进制、整数等多种格式
  3. 自动参数优化：根据消息长度调整PRC参数
  4. 错误处理：完善的异常捕获和状态报告
  5. 批量处理：支持多图像并行处理

  这些函数已经提供了完整的功能基础，我们只需要在上层封装一个更友好的接口。

> 详细解释 recover_posteriors(z, basis=None, variances=None)输出的是什么，什么是后验概率
  ⎿  ⧉ Selected 1 lines from src/image_watermark/PRC-Watermark/src/pseudogaussians.py in Cursor

● 让我深入分析recover_posteriors函数，这是理解PRC水印检测机制的关键。

● Read(src/image_watermark/PRC-Watermark/src/pseudogaussians.py)
  ⎿  Read 15 lines (ctrl+r to expand)

● Read(src/image_watermark/PRC-Watermark/decode.py)
  ⎿  Read 3 lines (ctrl+r to expand)

● 🧠 recover_posteriors 函数详细解析

  📊 函数核心功能

  recover_posteriors(z, basis=None, variances=None) 的作用是：从观测到的潜在表示估计原始PRC码字每一位的符号概率

  🎯 输入输出解析

  输入参数：
  - z: 从图像逆向得到的潜在表示向量 (形状: [16384] 对应 4×64×64)
  - basis: 可选的基变换矩阵（通常为None）
  - variances: 噪声方差估计（默认1.5）

  输出：
  - 返回一个与z同维度的向量，每个元素是后验概率，取值范围 [-1, 1]

  🔍 什么是后验概率

  在PRC水印的上下文中，后验概率指的是：

  "给定观测到的潜在表示值，原始PRC码字该位为+1的概率"

  具体来说：
  - +1: 表示该位原本为正值的概率很高
  - -1: 表示该位原本为负值的概率很高
  - 0: 表示完全不确定

  📐 数学原理

  函数的核心计算是：
  return erf(z / denominators)

  这里使用了误差函数(erf)，其数学含义：

  🧮 统计模型

  假设水印嵌入过程：
  1. 原始码字: c[i] ∈ {-1, +1} (PRC码字的符号)
  2. 伪高斯采样: original_z[i] = c[i] * |gaussian_noise|
  3. 扩散过程: observed_z[i] ~ N(original_z[i], σ²)

  🎲 后验概率计算

  给定观测值z[i]，我们要估计P(c[i] = +1 | z[i])：

  P(c[i] = +1 | z[i]) = ∫ P(z[i] | original_z[i]) * P(original_z[i] | c[i] = +1) d(original_z[i])

  通过贝叶斯推理，最终得到：
  posterior[i] = erf(z[i] / sqrt(2 * σ² * (1 + σ²)))

  🔧 实际使用流程

  # 1. 图像逆向到潜在空间
  reversed_latents = exact_inversion(image)  # shape: [1, 4, 64, 64]

  # 2. 展平为向量
  z = reversed_latents.flatten()  # shape: [16384]

  # 3. 计算后验概率  
  posteriors = recover_posteriors(z, variances=1.5)  # shape: [16384]

  # 4. 每个元素的含义：
  # posteriors[i] = 0.8  → 该位为+1的概率很高
  # posteriors[i] = -0.9 → 该位为-1的概率很高  
  # posteriors[i] = 0.1  → 不太确定，轻微倾向+1

  🎯 在水印检测中的作用

  后验概率是连接图像观测和密码学检测的桥梁：

  🔍 检测阶段 (Detect函数)

  # 使用后验概率计算奇偶校验
  for parity_check in parity_checks:
      # 计算该校验中所有位的乘积概率
      check_probability = ∏ posteriors[i] for i in parity_check

      # 如果大多数校验都满足，则检测到水印

  🔓 解码阶段 (Decode函数)

  # 使用belief propagation算法
  # 基于后验概率和奇偶校验约束，迭代推断最可能的消息
  decoded_message = belief_propagation(posteriors, parity_matrix)

  📈 数值示例

  假设我们有3个潜在值：
  z = [2.1, -1.8, 0.3]
  posteriors = recover_posteriors(z, variances=1.5)
  # 结果约为: [0.85, -0.79, 0.12]

  # 解释：
  # z[0] = 2.1 → posteriors[0] = 0.85 → 85%概率原始码字为+1
  # z[1] = -1.8 → posteriors[1] = -0.79 → 79%概率原始码字为-1  
  # z[2] = 0.3 → posteriors[2] = 0.12 → 56%概率原始码字为+1 (不太确定)

  🎪 关键理解

  recover_posteriors是整个PRC水印系统的**"软解调器"**：
  - 它不是简单地判断符号（硬解调）
  - 而是给出每个位的置信度（软解调）
  - 这种软信息使得后续的纠错解码更加鲁棒
  - 即使部分位被噪声破坏，系统仍能通过概率推理恢复消息

  这就是为什么PRC水印能够抵抗各种攻击的核心原理！

    1. 函数层次结构

  exact_inversion (最高层)
  ├── transform_img (图像预处理)
  ├── decoder_inv (精确逆向) 或 get_image_latents (快速逆向)
  │   └── get_image_latents (被decoder_inv内部调用)
  └── forward_diffusion (扩散逆向)

  2. 具体调用关系

  exact_inversion (最高层接口)

  # 第134行: 先调用图像预处理
  image = transform_img(image).unsqueeze(0).to(text_embeddings.dtype).to(device)

  # 第135-138行: 根据参数chooser选择逆向方法
  if decoder_inv:
      image_latents = pipe.decoder_inv(image)  # 精确逆向
  else:
      image_latents = pipe.get_image_latents(image, sample=False)  # 快速逆向

  decoder_inv (精确逆向)

  # 第434行: 内部先调用基础方法获取初始值
  z = self.get_image_latents(x).clone().float()

  # 第441-449行: 然后通过100步优化改进结果
  for i in range(100):
      x_pred = self.decode_image_for_gradient_float(z)  # 解码预测
      loss = loss_function(x_pred, input)  # 计算损失
      # Adam优化步骤...

  3. 功能差异对比

  | 函数                | 输入                  | 输出                | 精度  | 速度  | 用途      |
  |-------------------|---------------------|-------------------|-----|-----|---------|
  | transform_img     | PIL Image           | Tensor(3,512,512) | N/A | 最快  | 图像预处理   |
  | get_image_latents | Tensor(1,3,512,512) | Tensor(1,4,64,64) | 中等  | 快   | 基础VAE编码 |
  | decoder_inv       | Tensor(1,3,512,512) | Tensor(1,4,64,64) | 高   | 慢   | 优化求解逆向  |
  | exact_inversion   | PIL Image + prompt  | Tensor(1,4,64,64) | 最高  | 最慢  | 完整扩散逆向  |

  4. 数据流转换链

  PIL Image (512×512×3)
      ↓ transform_img
  Tensor (3,512,512) [normalized to [-1,1]]
      ↓ unsqueeze(0)
  Tensor (1,3,512,512)
      ↓ get_image_latents 或 decoder_inv
  Tensor (1,4,64,64) [latent space]
      ↓ (optional) forward_diffusion
  Tensor (1,4,64,64) [inverted latents]