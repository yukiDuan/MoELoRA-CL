# ==============================
# MoELoRA 持续学习实验 - 主脚本
# ==============================
import torch

from utils import (
    DEFAULT_CONFIG,
    load_model_and_tokenizer,
    create_continual_learning_tasks,
    make_dataloader,
)
from modeling import MoELoRAModel
from trainer import MoELoRATrainer, ForgettingMonitor


def main():
    config = DEFAULT_CONFIG
    print("=" * 50)
    print("MoELoRA 持续学习实验")
    print("=" * 50)
    print(f"配置: {config}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # --------------------------
    # 1. 加载基座模型
    # --------------------------
    print("\n[1/6] 加载基座模型...")
    base_model, tokenizer = load_model_and_tokenizer(config["model_name"])

    # --------------------------
    # 2. 构建 MoELoRA 模型
    # --------------------------
    print("\n[2/6] 构建 MoELoRA 模型...")
    model = MoELoRAModel(
        base_model=base_model,
        target_modules=config["target_modules"],
        num_experts=config["num_experts"],
        r=config["lora_r"],
        alpha=config["lora_alpha"],
        top_k=config["top_k"],
    )
    model.print_trainable_parameters()

    # --------------------------
    # 3. 准备持续学习数据
    # --------------------------
    print("\n[3/6] 准备持续学习数据...")
    tasks = create_continual_learning_tasks(tokenizer, config["max_length"])
    task_configs = []
    for task in tasks:
        task_configs.append({
            "name": task["name"],
            "train_loader": make_dataloader(
                task["train_data"], config["batch_size"], shuffle=True
            ),
            "eval_loader": make_dataloader(
                task["eval_data"], config["batch_size"], shuffle=False
            ),
        })
        print(f"  {task['name']}: train={len(task['train_data'])} eval={len(task['eval_data'])}")

    # --------------------------
    # 4. 初始化训练器和遗忘监控
    # --------------------------
    print("\n[4/6] 初始化训练器...")
    monitor = ForgettingMonitor(task_names=[t["name"] for t in tasks])
    trainer = MoELoRATrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        lr=config["lr"],
        lambda_lb=config["lambda_lb"],
        lambda_cl=config["lambda_cl"],
    )

    # --------------------------
    # 5. 顺序训练所有任务
    # --------------------------
    print("\n[5/6] 开始持续学习训练...")
    trainer.train_continual(
        task_configs=task_configs,
        monitor=monitor,
        num_epochs=config["num_epochs"],
    )

    # --------------------------
    # 6. 遗忘分析报告
    # --------------------------
    print("\n[6/6] 遗忘分析报告")
    print(monitor.summary())

    # --------------------------
    # 7. 推理验证
    # --------------------------
    print("\n推理验证:")
    model.eval()
    model.base_model.config.use_cache = True
    prompts = [
        "Translate the following to Chinese: Nice weather today. Answer:",
        "Question: What is LoRA? Answer:",
        "Summarize: Neural networks learn from data. Answer:",
    ]
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.base_model.generate(
                **inputs, max_new_tokens=30, do_sample=False
            )
        print(f"  输入: {prompt}")
        print(f"  输出: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        print()


if __name__ == "__main__":
    main()
