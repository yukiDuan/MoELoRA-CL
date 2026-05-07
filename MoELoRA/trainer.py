# ==============================
# MoELoRA 训练器 + 遗忘监控
# ==============================
import math
from collections import defaultdict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from modeling import MoELoRAModel, contrastive_loss_placeholder


# --------------------------
# 1. 遗忘监控器
# --------------------------
class ForgettingMonitor:

    def __init__(self, task_names: list[str]):
        self.task_names = list(task_names)
        self.history: dict[str, list[dict]] = defaultdict(list)

    def record(self, task_name: str, stage: str, metrics: dict):
        entry = {"stage": stage, **metrics}
        self.history[task_name].append(entry)

    def compute_forgetting(self) -> dict:
        """
        BWT (Backward Transfer): 衡量学完所有任务后旧任务的性能退化
        BWT = (1/(T-1)) * Σ_{i=1}^{T-1} (R_final_i - R_best_i)
        负值表示遗忘，越接近 0 越好
        """
        results = {}
        for task_name in self.task_names:
            records = self.history.get(task_name, [])
            if len(records) < 2:
                continue
            best_loss = min(r["loss"] for r in records)
            final_loss = records[-1]["loss"]
            results[task_name] = {
                "best_loss": best_loss,
                "final_loss": final_loss,
                "forgetting": final_loss - best_loss,
            }
        return results

    def summary(self) -> str:
        lines = ["\n" + "=" * 50, "遗忘监控报告", "=" * 50]

        for task_name in self.task_names:
            records = self.history.get(task_name, [])
            lines.append(f"\n[{task_name}]")
            for r in records:
                ppl = r.get("perplexity", float("nan"))
                lines.append(f"  {r['stage']:30s}  loss={r['loss']:.4f}  ppl={ppl:.2f}")

        forgetting = self.compute_forgetting()
        if forgetting:
            lines.append("\n--- 遗忘指标 (loss 增量，越小越好) ---")
            for task_name, info in forgetting.items():
                lines.append(
                    f"  {task_name}: best={info['best_loss']:.4f} "
                    f"final={info['final_loss']:.4f} "
                    f"forgetting={info['forgetting']:+.4f}"
                )

        lines.append("=" * 50)
        return "\n".join(lines)


# --------------------------
# 2. MoELoRA 训练器
# --------------------------
class MoELoRATrainer:

    def __init__(
        self,
        model: MoELoRAModel,
        tokenizer,
        device: str = "cuda",
        lr: float = 2e-4,
        lambda_lb: float = 0.01,
        lambda_cl: float = 0.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.lambda_lb = lambda_lb
        self.lambda_cl = lambda_cl

        self.model.to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=lr)

    def train_single_task(
        self,
        task_name: str,
        train_loader: DataLoader,
        eval_loader: DataLoader = None,
        num_epochs: int = 3,
        monitor: "ForgettingMonitor" = None,
        all_eval_loaders: dict[str, DataLoader] = None,
    ):
        print(f"\n{'='*40}")
        print(f"开始训练任务: {task_name}")
        print(f"{'='*40}")

        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            steps = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                loss, loss_dict = self._train_step(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()
                total_loss += loss_dict["task_loss"]
                steps += 1

            avg_loss = total_loss / max(steps, 1)
            print(f"  Epoch {epoch+1}/{num_epochs}  avg_task_loss={avg_loss:.4f}")

            if monitor and all_eval_loaders:
                self._eval_all_tasks(
                    all_eval_loaders, monitor,
                    stage=f"after_{task_name}_epoch{epoch+1}",
                )

    def train_continual(
        self,
        task_configs: list[dict],
        monitor: "ForgettingMonitor" = None,
        num_epochs: int = 3,
    ):
        all_eval_loaders = {
            cfg["name"]: cfg["eval_loader"]
            for cfg in task_configs
            if cfg.get("eval_loader") is not None
        }

        if monitor and all_eval_loaders:
            self._eval_all_tasks(all_eval_loaders, monitor, stage="before_training")

        for cfg in task_configs:
            self.train_single_task(
                task_name=cfg["name"],
                train_loader=cfg["train_loader"],
                eval_loader=cfg.get("eval_loader"),
                num_epochs=num_epochs,
                monitor=monitor,
                all_eval_loaders=all_eval_loaders,
            )

    def _train_step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        task_loss = outputs.loss

        lb_loss = self.model.get_load_balance_loss()

        task_ids = torch.zeros(batch["input_ids"].shape[0], device=self.device)
        router_dists = self.model.get_router_distributions()
        cl_loss = contrastive_loss_placeholder(router_dists, task_ids)

        total_loss = task_loss + self.lambda_lb * lb_loss + self.lambda_cl * cl_loss

        return total_loss, {
            "task_loss": task_loss.item(),
            "lb_loss": lb_loss.item(),
            "cl_loss": cl_loss.item(),
        }

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for batch in eval_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            labels = batch["labels"]
            num_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
        self.model.train()

        avg_loss = total_loss / max(total_tokens, 1)
        return {"loss": avg_loss, "perplexity": math.exp(min(avg_loss, 20))}

    def _eval_all_tasks(
        self,
        all_eval_loaders: dict[str, DataLoader],
        monitor: ForgettingMonitor,
        stage: str,
    ):
        for task_name, loader in all_eval_loaders.items():
            metrics = self.evaluate(loader)
            monitor.record(task_name, stage, metrics)
            print(
                f"    [eval] {task_name:20s} loss={metrics['loss']:.4f} "
                f"ppl={metrics['perplexity']:.2f}  ({stage})"
            )
