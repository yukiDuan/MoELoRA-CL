# ==============================
# MoELoRA 模型定义
# LoRA Expert / MoELoRA Layer / MoELoRA Model
# ==============================
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


# --------------------------
# 1. 单个 LoRA 专家
# --------------------------
class LoRAExpert(nn.Module):

    def __init__(self, d_in: int, d_out: int, r: int = 8, alpha: float = 16.0, dtype=None):
        super().__init__()
        self.lora_A = nn.Linear(d_in, r, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(r, d_out, bias=False, dtype=dtype)
        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


# --------------------------
# 2. MoELoRA 层（替换单个 nn.Linear）
# --------------------------
class MoELoRALayer(nn.Module):

    def __init__(
        self,
        base_linear: nn.Linear,
        num_experts: int = 4,
        r: int = 8,
        alpha: float = 16.0,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        d_in = base_linear.in_features
        d_out = base_linear.out_features

        self.base_linear = base_linear
        for param in self.base_linear.parameters():
            param.requires_grad = False

        dtype = base_linear.weight.dtype
        self.experts = nn.ModuleList(
            [LoRAExpert(d_in, d_out, r, alpha, dtype=dtype) for _ in range(num_experts)]
        )
        self.router = nn.Linear(d_in, num_experts, bias=False, dtype=dtype)

        self.last_load_balance_loss = torch.tensor(0.0)
        self.last_router_probs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)

        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        N = x_flat.shape[0]

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        self.last_router_probs = router_probs.detach()

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        self._compute_load_balance_loss(router_probs, top_k_indices)

        expert_out = torch.zeros(N, base_out.shape[-1], device=x.device, dtype=x.dtype)
        for k_idx in range(self.top_k):
            idx = top_k_indices[:, k_idx]
            gate = top_k_probs[:, k_idx].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = idx == e
                if mask.any():
                    expert_out[mask] += gate[mask] * self.experts[e](x_flat[mask])

        expert_out = expert_out.reshape(orig_shape[:-1] + (base_out.shape[-1],))
        return base_out + expert_out

    def _compute_load_balance_loss(
        self, router_probs: torch.Tensor, top_k_indices: torch.Tensor
    ):
        N = router_probs.shape[0]
        P = router_probs.mean(dim=0)
        one_hot = F.one_hot(top_k_indices, self.num_experts).float().sum(dim=1)
        f = one_hot.mean(dim=0)
        self.last_load_balance_loss = (self.num_experts * (f * P).sum())



# --------------------------
# 3. MoELoRA 模型（包装基座模型）
# --------------------------
def _get_parent_module(model: nn.Module, dotted_name: str):
    parts = dotted_name.split(".")
    current = model
    for part in parts[:-1]:
        current = getattr(current, part)
    return current, parts[-1]


class MoELoRAModel(nn.Module):

    def __init__(
        self,
        base_model: PreTrainedModel,
        target_modules: list[str],
        num_experts: int = 4,
        r: int = 8,
        alpha: float = 16.0,
        top_k: int = 2,
    ):
        super().__init__()
        self.base_model = base_model

        for param in self.base_model.parameters():
            param.requires_grad = False

        self._replace_with_moelora(target_modules, num_experts, r, alpha, top_k)

        if hasattr(self.base_model, "enable_input_require_grads"):
            self.base_model.enable_input_require_grads()

    def _replace_with_moelora(self, target_modules, num_experts, r, alpha, top_k):
        replaced = []
        for name, module in list(self.base_model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in target_modules):
                continue
            parent, attr = _get_parent_module(self.base_model, name)
            moelora_layer = MoELoRALayer(module, num_experts, r, alpha, top_k)
            setattr(parent, attr, moelora_layer)
            replaced.append(name)
        print(f"[MoELoRA] 替换了 {len(replaced)} 个线性层: {replaced[:4]}...")

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    def get_load_balance_loss(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        for module in self.modules():
            if isinstance(module, MoELoRALayer):
                total = total + module.last_load_balance_loss
                count += 1
        return total / max(count, 1)

    def get_router_distributions(self) -> dict[str, torch.Tensor]:
        dists = {}
        for name, module in self.named_modules():
            if isinstance(module, MoELoRALayer) and module.last_router_probs is not None:
                dists[name] = module.last_router_probs
        return dists

    def print_trainable_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100.0 * trainable / total if total > 0 else 0
        print(
            f"可训练参数: {trainable:,} / {total:,} ({pct:.2f}%)"
        )

    def get_expert_params(self) -> list[nn.Parameter]:
        params = []
        for module in self.modules():
            if isinstance(module, LoRAExpert):
                params.extend(module.parameters())
        return params

    def get_router_params(self) -> list[nn.Parameter]:
        params = []
        for module in self.modules():
            if isinstance(module, MoELoRALayer):
                params.extend(module.router.parameters())
        return params


# --------------------------
# 4. 对比学习损失占位（InfoNCE）
# --------------------------
def contrastive_loss_placeholder(
    router_dists: dict[str, torch.Tensor],
    task_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE 对比损失占位。
    完整实现：同任务样本为正对，不同任务为负对，作用于路由分布。
    当前返回 0，保留接口供后续实现。
    """
    device = task_ids.device if isinstance(task_ids, torch.Tensor) else "cpu"
    return torch.tensor(0.0, device=device, requires_grad=True)
