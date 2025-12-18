# 🌌 H2Q-MicroStream: Holographic Hamiltonian Quaternion Transformer

> **"智能不是记忆过去的所有细节，而是掌握生成未来的核心方程。"**
>
> **"Intelligence is not about memorizing every detail of the past, but mastering the core equations that generate the future."**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Experimental-blueviolet)](https://github.com/)

---

## 📖 项目简介 / Introduction

**H2Q-MicroStream** 是一个极具实验性的深度学习架构，旨在探索**语言模型的物理动力学本质**。与追求巨大参数量和超长上下文窗口的主流 Transformer 不同，本项目基于**奥卡姆剃刀原则 (Occam's Razor)** 和 **全息原理 (Holographic Principle)**，构建了一个极简、实时、且具有强物理约束的“思维内核”。

**H2Q-MicroStream** is a highly experimental deep learning architecture designed to explore the **physical dynamics of language models**. Unlike mainstream Transformers that chase massive parameter counts and infinite context windows, this project builds a minimalist, real-time, and physically constrained "Thinking Kernel" based on **Occam's Razor** and the **Holographic Principle**.

### 核心哲学 / Core Philosophy

1.  **思考内化 vs. 语言表达 (Internalization vs. Expression)**:
    *   我们认为，现有的 LLM 花费了太多算力去学习“如何像人一样说话”（语法糖），而忽略了“如何构建世界模型”（核心逻辑）。
    *   H2Q 旨在构建一个**高维全向的思维核心**。它的中间状态可能人类难以直接理解（类似于脑电波），但它包含了对信息本质的拓扑映射。
    *   *We believe current LLMs spend too much compute on "speaking like a human" (syntax) rather than "modeling the world" (core logic). H2Q aims to build a high-dimensional, omnidirectional thinking kernel.*

2.  **状态保持 vs. 历史回溯 (State-based vs. Retrieval-based)**:
    *   人类没有 128k 的上下文窗口。我们靠的是**核心状态 (State)** 的实时演化。
    *   本架构放弃了对历史数据的无限 Attention，转而追求在极短视界（Micro-Horizon）内的**哈密顿动力学演化**。
    *   *Humans don't utilize 128k context windows; we rely on the real-time evolution of a Core State. This architecture abandons infinite attention on history in favor of Hamiltonian dynamic evolution within a Micro-Horizon.*

3.  **本质压缩 (Essence Compression)**:
    *   如果一个规律不能用极少的基底（Rank 8）解释，那就是在死记硬背。
    *   *If a pattern cannot be explained with a minimal basis (Rank 8), it is rote memorization, not learning.*

---

## 🚀 关键技术特性 / Key Technical Features

### 1. 🌌 四元数时空注意力 (Quaternion Spacetime Attention)
引入**四元数 (Quaternion)** 代数，将注意力机制从标量积升级为**四维时空干涉**。
*   **实部 (Real Part)**: 代表能量/幅度，决定注意力的强度。
*   **虚部 (Imaginary Part)**: 代表自旋/相位，引入非线性的**相位旋转反馈 (Phase Rotation)**。
*   这使得模型能够捕捉语言中的“纠缠”和“反讽”等高维特征。

*Moves attention from scalar products to **4D spacetime interference**. Real parts represent energy/amplitude; Imaginary parts represent spin/phase, introducing nonlinear Phase Rotation Feedback to capture high-dimensional linguistic entanglement.*

### 2. 📉 Rank-8 本质约束 (Rank-8 Essential Constraint)
模型权重不是静态矩阵，而是通过 **Structure Bank** 动态生成的。我们强制将 Rank 限制为 **8**。
*   这逼迫模型放弃“背书”，只能提取最核心的 8 种时空演化规律。
*   这也极大地降低了计算消耗，实现了参数的“全息折叠”。

*Weights are dynamically generated via a Structure Bank with a forced **Rank of 8**. This forces the model to abandon rote memorization and extract only the 8 most essential spacetime evolution patterns.*

### 3. 🌊 Unicode 流式动力学 (Unicode Stream Dynamics)
摒弃了 BPE Tokenizer（如 Tiktoken），直接使用 **Unicode (ASCII/UTF-8)** 编码。
*   **拒绝“方言”**：建立通用的底层物理接口，让模型直接处理字节流。
*   **并行流训练**：模拟多路并行的连续阅读体验，而非随机切片。

*Abandons BPE Tokenizers for direct **Unicode (ASCII/UTF-8)** encoding. establishing a universal physical interface. Uses parallel streaming to simulate continuous reading flow rather than random slicing.*

### 4. ⚡️ 微批次高频更新 (Micro-Batch High-Freq Update)
*   **Batch Size = 24**: 模拟极低容量的短期记忆。
*   **No Gradient Accumulation**: 每看一眼数据就更新一次参数。
*   这模拟了生物神经元的**高频脉冲学习**，使参数在流形空间中进行连续的微分演化。

*Simulates biological high-frequency impulse learning. With a micro-batch of 24 and continuous updates, the parameters undergo continuous differential evolution in the manifold space.*

---

## 🛠️ 安装与运行 / Installation & Usage

### 环境要求 / Requirements
*   Python 3.8+
*   PyTorch 2.0+ (CUDA support recommended for TF32 acceleration)
*   NVIDIA GPU (Optimized for Ampere/Ada architectures like RTX 3090/4090/4070Ti)

### 快速开始 / Quick Start

1.  **克隆仓库 / Clone the repository**
    ```bash
    git clone https://github.com/makai891124-prog/H2Q-Transformer.git
    cd H2Q-Transformer
    ```

2.  **安装依赖 / Install dependencies**
    ```bash
    pip install torch numpy requests
    ```

3.  **运行训练 / Run training**
    无需手动下载数据，脚本会自动下载 WikiText-2 数据集并开始训练。
    *No need to manually download data; the script will automatically download WikiText-2 and start training.*
    ```bash
    python main.py
    ```

---

## 📊 配置说明 / Configuration

在 `main.py` 中的 `CONFIG` 字典中调整参数。当前默认配置为 **"H2Q-MicroStream"** 模式：

```python
CONFIG = {
    'dim': 768,            # 模型宽度 (GPT-2 Small level)
    'fixed_rank': 8,       # 🌟 核心参数：限制模型的"脑容量"以逼迫其思考
    'seq_len': 128,        # 微视界：只关注当下瞬间
    'batch_size': 24,      # 物理 Batch：极小，高频更新
    'depth': 12,           # 深度
    'axiom_lambda': 0.1,   # 正交性约束强度
    # ...
}
```

---

## 🔮 展望与未来 / Future Roadmap

目前的 H2Q 模型是一个**纯粹的思维内核**。它的输出可能看起来像“乱码”或极其抽象的方言，这是因为它正在展示内部的**原始状态流**。

未来的开发计划包括：
1.  **解码器挂载 (Projector)**: 训练一个独立的“翻译器”模块，将 H2Q 的全息状态映射回人类自然语言。
2.  **多模态流 (Multimodal Stream)**: 由于采用 Unicode/Byte 接口，尝试直接输入音频或图像字节流。
3.  **边缘侧部署 (Edge Deployment)**: 利用 Rank-8 的极高压缩率，尝试在移动端运行全息内核。

*The current H2Q model is a **pure thinking kernel**. Future plans include training a separate "Projector" to translate holographic states into human language, exploring multimodal byte streams, and edge deployment via high compression rates.*

---

## 📜 许可证 / License

本项目采用 [MIT License](LICENSE) 开源。

---

### 致谢 / Acknowledgements
感谢所有探索几何深度学习、SSM (State Space Models) 以及对 Transformer 架构进行反思的研究者们。本项目的灵感来源于全息原理、哈密顿力学以及人类认知的本质。
