# ==============================
# MoELoRA 辅助函数
# 环境配置 / 数据加载 / 工具函数
# ==============================
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator


# --------------------------
# 1. 默认配置
# --------------------------
DEFAULT_CONFIG = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "target_modules": ["q_proj", "v_proj"],
    "num_experts": 4,
    "top_k": 2,
    "lora_r": 8,
    "lora_alpha": 16.0,
    "lr": 2e-4,
    "num_epochs": 3,
    "batch_size": 4,
    "max_length": 128,
    "lambda_lb": 0.01,
    "lambda_cl": 0.0,
}


# --------------------------
# 2. 模型加载
# --------------------------
def load_model_and_tokenizer(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    return model, tokenizer


# --------------------------
# 3. 持续学习任务数据
# --------------------------
TASK_DATA = {
    "task_0_translation": [
        "Translate the following to Chinese: Hello world. Answer: 你好世界。",
        "Translate the following to Chinese: Good morning. Answer: 早上好。",
        "Translate the following to Chinese: Thank you very much. Answer: 非常感谢。",
        "Translate the following to Chinese: How are you? Answer: 你好吗？",
        "Translate the following to Chinese: See you tomorrow. Answer: 明天见。",
        "Translate the following to Chinese: I love programming. Answer: 我喜欢编程。",
    ],
    "task_1_qa": [
        "Question: What is LoRA? Answer: LoRA is a parameter-efficient fine-tuning method using low-rank decomposition.",
        "Question: What is continual learning? Answer: Continual learning enables models to learn new tasks without forgetting old ones.",
        "Question: What is MoE? Answer: Mixture of Experts routes inputs to specialized sub-networks via a gating mechanism.",
        "Question: What is catastrophic forgetting? Answer: It is the tendency of neural networks to forget previously learned tasks when trained on new ones.",
        "Question: What is a router in MoE? Answer: A router is a gating network that decides which experts process each input token.",
        "Question: What is knowledge distillation? Answer: It is a technique where a smaller model learns to mimic a larger model's behavior.",
    ],
    "task_2_summarization": [
        "Summarize: Large language models have shown remarkable capabilities in various NLP tasks. Answer: LLMs excel at diverse NLP tasks.",
        "Summarize: Parameter-efficient fine-tuning reduces the cost of adapting large models to downstream tasks. Answer: PEFT makes LLM adaptation cheaper.",
        "Summarize: Mixture of experts allows scaling model capacity without proportionally increasing computation. Answer: MoE scales capacity efficiently.",
        "Summarize: Continual learning aims to learn sequentially without forgetting prior knowledge. Answer: CL learns sequentially while retaining knowledge.",
        "Summarize: Contrastive learning pulls similar representations together and pushes dissimilar ones apart. Answer: Contrastive learning shapes representation geometry.",
        "Summarize: The transformer architecture uses self-attention to model long-range dependencies. Answer: Transformers capture long-range dependencies via attention.",
    ],
}


def create_continual_learning_tasks(tokenizer, max_length: int = 128) -> list[dict]:
    tasks = []
    for task_name, raw_texts in TASK_DATA.items():
        train_texts = raw_texts[:4]
        eval_texts = raw_texts[4:]
        tasks.append({
            "name": task_name,
            "train_data": tokenize_data(train_texts, tokenizer, max_length),
            "eval_data": tokenize_data(eval_texts, tokenizer, max_length),
        })
    return tasks


# --------------------------
# 4. 分词与数据处理
# --------------------------
def tokenize_data(raw_texts: list[str], tokenizer, max_length: int = 128) -> list[dict]:
    features = []
    for text in raw_texts:
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        inputs["labels"] = [
            -100 if token == tokenizer.pad_token_id else token
            for token in inputs["input_ids"]
        ]
        features.append(inputs)
    return features


def make_dataloader(
    features: list[dict], batch_size: int = 4, shuffle: bool = True
) -> DataLoader:
    return DataLoader(
        features,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=default_data_collator,
    )


# --------------------------
# 5. 工具函数
# --------------------------
def print_model_modules(model):
    """打印模型所有模块名称，用于确认 target_modules"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"  {name}: Linear({module.in_features}, {module.out_features})")
