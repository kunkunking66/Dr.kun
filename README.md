# Dr. Kun: 中文医疗专家大模型微调实战教程

欢迎来到 Dr. Kun 项目！本仓库完整记录了如何使用 **Unsloth** 工具库，对 `DeepSeek-R1-Distill-Llama-8B` 这样优秀的开源大模型进行高效微调（Fine-Tuning），使其从一个通用模型转变为一个专注于中文医疗问答的专家模型。

这个教程不仅是项目的代码记录，更是一份详尽的、从零开始的 LLM 微调指南，希望能帮助您快速掌握大模型微调的核心技术。

## 项目亮点

*   **🚀 高效训练**：借助 Unsloth，微调速度提升 **2倍**，显存占用降低高达 **80%**，即使在消费级显卡（如 Colab 的 T4）上也能轻松完成。
*   **🧠 专家能力**：通过在专业医疗问答数据集上进行监督式微调（SFT），模型学会了像医学专家一样，进行有逻辑、有条理的思考和回答。
*   **💼 部署友好**：最终将模型转换为 **GGUF** 格式，这是一种高度量化的模型文件，可以在没有高端 GPU 的情况下，甚至在 CPU 上高效运行，极大地方便了本地部署和应用。
*   **📖 端到端流程**：教程覆盖了从环境配置、模型加载、数据处理、模型训练，到最终模型保存和上传的完整流程。

---

## 微调流程详解

接下来，我们将一步步拆解 `Dr.kun.ipynb` 中的核心代码，详细解释每一步的目的和关键操作。

### 第一步：环境准备与依赖安装

一个稳定可靠的运行环境是成功的开始。首先，我们需要安装本次项目所需的所有核心库。

```python
# %%capture
# 註：%%capture 是 Colab/Jupyter 的魔法命令，它会隐藏单元格的输出，让笔记本更整洁。

# 1. 安装 Unsloth 核心库
# Unsloth 是我们本次微调的加速引擎。
!pip install unsloth

# 2. 从 GitHub 安装最新版本的 Unsloth
# 为了确保能用上最新的优化和功能，我们直接从源代码进行安装，这是非常好的习惯。
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

# 3. 安装 bitsandbytes
# 这个库让模型量化成为可能，是降低显存占用的关键技术。
!pip install bitsandbytes
```

### 第二步：加载基础模型

我们选择 `unsloth/DeepSeek-R1-Distill-Llama-8B` 作为基础模型，它性能强劲且对硬件要求相对友好。

```python
from unsloth import FastLanguageModel
import torch

# --- 模型配置 ---
max_seq_length = 2048  # 模型能处理的最大文本长度
dtype = None  # 数据类型，设为 None 让 Unsloth 自动检测最佳配置
load_in_4bit = True  # 核心！启用 4-bit 量化加载，极大降低显存需求

# --- 从 Hugging Face 加载模型和分词器 ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```
*   **`load_in_4bit = True`**: 这是 Unsloth 的魔法所在。它意味着模型的参数将以 4-bit 的精度加载到显存中，相比传统的 16-bit，显存占用理论上可以降低 **75%**！

### 第三步：微调前性能测试（设定基准）

在“教”模型新知识之前，我们先看看它原本的水平如何。这有助于我们评估微调带来的巨大提升。

```python
# 定义一个符合医疗专家角色的提问模板
prompt_style = """...
### 指令：
你是一名医学领域的专家，精通各种疑难杂症
请回答以下医学问题。
### 问题：
{}
### 回答：
<think>{}"""

question = "一名 48 岁男性患者...应优先选用哪种药物进行治疗？"

# 将模型切换到推理模式并进行提问
FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors = "pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens = 1200, use_cache = True)
print(tokenizer.batch_decode(outputs))
```
*   **结果分析**：未经微调的通用模型在回答专业问题时，往往会给出宽泛、甚至不准确的答案。这为我们的微调效果提供了清晰的对比基准。

### 第四步：准备与格式化训练数据

数据是模型的“精神食粮”。我们使用 `FreedomIntelligence/medical-o1-reasoning-SFT` 数据集，它包含大量带有“思考过程（Chain-of-Thought）”的医疗问答。

```python
# 这是为训练数据设计的格式，包含“思考过程”
train_prompt_style = """...
### 回答：
<思考>
{}
</思考>
{}"""

# 加载数据集
from datasets import load_dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", 'zh', split = "train[0:5000]")

# 定义一个函数，将原始数据转换为我们的训练格式
def formatting_prompts_func(examples):
    inputs       = examples["Question"]
    cots         = examples["Complex_CoT"]
    outputs      = examples["Response"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

# 将格式化函数应用到整个数据集
dataset = dataset.map(formatting_prompts_func, batched = True)
```
*   **关键点**：让模型学习“思考过程 (`<思考>...</思考>`)”至关重要。这能教会模型如何进行逻辑推理，而不仅仅是给出最终答案，从而极大提升回答的质量和可靠性。

### 第五步：使用 LoRA 进行高效微调

如果说微调是给模型做“手术”，那么 **LoRA**（Low-Rank Adaptation）就是一种“微创手术”。它无需改动模型原有的数十亿个巨大参数，而是通过在模型的关键部分（如注意力层）旁边添加少量可训练的“适配器”（Adapter）层来学习新知识，极大地降低了训练成本。

```python
# 让模型为 LoRA 训练做好准备
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA 的秩(rank)。数值越高，可训练参数越多，通常 8, 16, 32 即可。
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"], # 指定要应用 LoRA 的模型层。这些通常是注意力层和前馈网络层。
    lora_alpha = 16,  # LoRA 的缩放因子，通常设置为与 r 相同或两倍。
    lora_dropout = 0,  # Dropout 比率，用于防止过拟合，在小数据集上可以设为0。
    bias = "none",  # 是否训练偏置项，"none" 表示不训练。
    use_gradient_checkpointing = "unsloth",  # 使用 Unsloth 优化的梯度检查点技术，极大节省显存。
    random_state = 3407,
)
```

### 第六步：配置并启动训练器

万事俱备，只欠东风。我们使用 Hugging Face 的 `SFTTrainer` 来配置训练过程中的所有超参数（学习率、批大小、步数等），然后启动训练。

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,    # 每个 GPU 的批大小。
        gradient_accumulation_steps = 4,    # 梯度累积步数。最终的批大小 = 2 * 4 = 8，以此模拟更大批次的训练。
        warmup_steps = 5,                   # 学习率预热步数，在训练初期逐步增加学习率。
        max_steps = 75,                     # 设定总的训练步数。
        learning_rate = 2e-4,               # 学习率，决定了模型参数更新的幅度。
        fp16 = not is_bfloat16_supported(), # 自动检测并启用半精度训练，加速计算。
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,                  # 每隔一步就记录一次训练日志（如loss）。
        optim = "adamw_8bit",               # 使用 8-bit 优化器，进一步节省显存。
        weight_decay = 0.01,                # 权重衰减，一种正则化手段，防止过拟合。
        lr_scheduler_type = "linear",       # 学习率调度策略。
        seed = 3407,                        # 随机种子，保证实验可复现。
        output_dir = "outputs",             # 保存检查点和日志的目录。
    ),
)

# 启动训练！
trainer_stats = trainer.train()
```
*   **训练监控**：在训练过程中，密切关注 `Training Loss` 的变化。一个理想的训练过程，loss 值应该会稳步下降，这表明模型正在有效地从数据中学习。

### 第七步：微调后效果评估

“手术”完成了，现在是见证奇迹的时刻。我们用与第三步完全相同的代码和问题再次提问，看看模型的表现是否脱胎换骨。

```python
# 同样的问题，不同的模型（现在是微调过的模型）
FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 4000, use_cache = True)
print(tokenizer.batch_decode(outputs))
```
*   **效果对比**：你会惊喜地发现，微调后的模型回答变得极为专业、详细，并且包含了逻辑清晰的思考过程。它不再是一个泛泛而谈的通用模型，而是一个名副其实的“Dr. Kun”。

### 第八步：模型保存与转换为 GGUF

为了让模型能够方便地在各种设备上（甚至是只有 CPU 的普通电脑）部署，我们将其转换为 **GGUF** 格式。GGUF 是一种为 `llama.cpp` 等推理框架设计的、高度优化的量化模型文件。

```python
# 将 LoRA 适配器与基础模型权重合并，然后量化并保存为 GGUF 格式
# "Q8_0" 代表 8-bit 量化，在模型大小和性能之间取得了很好的平衡。
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q8_0")
```
*   **`save_pretrained_gguf`**: 这是 Unsloth 提供的一键转换功能，它会自动完成合并 LoRA 权、量化、转换格式等所有复杂步骤，最终在 `model` 文件夹下生成一个 `unsloth.Q8_0.gguf` 文件。

### 第九步：上传到 Hugging Face Hub

最后，将你训练好的模型分享给全世界，或者方便自己在其他地方调用！

```python
from huggingface_hub import create_repo, HfApi

# 登录 Hugging Face (需要提前在 Colab Secrets 中设置 HUGGINGFACE_TOKEN)
from google.colab import userdata
HF_TOKEN = userdata.get('HUGGINGFACE_TOKEN')
api = HfApi()

# 你的 Hugging Face 用户名和仓库名
repo_id = "你的用户名/Dr.kun" 

# 在 Hugging Face Hub 上创建一个新的模型仓库
create_repo(repo_id, exist_ok=True, token=HF_TOKEN)

# 上传 GGUF 模型文件到你的仓库
api.upload_file(
    path_or_fileobj="model/unsloth.Q8_0.gguf",
    path_in_repo="unsloth.Q8_0.gguf",
    repo_id=repo_id,
    token=HF_TOKEN,
)
```
*   **分享成果**：上传成功后，任何人都可以通过 Hugging Face 访问你的模型主页，下载这个 `.gguf` 文件，并使用 `llama.cpp`、`Ollama` 等工具在本地轻松运行它。

---

## 拓展技能：玩转 Hugging Face

Hugging Face 是当今 AI 领域最重要的社区和平台，被誉为“AI 界的 GitHub”。学会使用它，你就能轻松获取海量的模型和数据资源。

### 如何为你的领域寻找数据集？

假设你想微调一个**金融领域**的问答模型，可以遵循以下步骤：

1.  **访问数据集页面**: 打开 [Hugging Face Datasets](https://huggingface.co/datasets)。
2.  **使用关键词搜索**: 在搜索框中输入与你领域相关的关键词，如 "finance", "金融", "法律", "用户评论" 等。
3.  **利用筛选器精准定位**: 在左侧筛选栏中，选择 **Tasks (任务)** (如 `Question Answering`) 和 **Languages (语言)** (如 `Chinese (zh)`)。
4.  **评估与预览数据集**: 点击进入数据集页面，阅读 **Dataset Card (数据集卡片)** 了解详情，并使用 **Viewer** 功能在线预览数据格式。
5.  **在代码中一键加载**: 找到心仪的数据集后，复制其名称，用 `load_dataset("数据集名称", split="train")` 即可在你的代码中加载并使用。

通过以上步骤，你就可以为任何垂直领域找到合适的训练数据，并复用本项目的微调流程，训练出属于你自己的专家模型！
