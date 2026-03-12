import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Set, Tuple, Optional, Any, Dict, List

import torch.nn as nn
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from xtuner.v1.loss import CELossContext
from xtuner.v1.model.base import BaseModel as XTunerBaseModel
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.model.compose.base import XTunerBaseModelConfig
from xtuner.v1.model.moe.moe import MoEModelOutputs, SequenceContext
from xtuner.v1.model.utils.misc import update_weight_map_from_safetensors_index
from xtuner.v1.utils import get_logger, profile_time_and_memory
from pydantic import ConfigDict, BaseModel, Field

try:
    from xtuner.v1.float8 import prepare_float8_for_fsdp
except ImportError:
    prepare_float8_for_fsdp = None

logger = get_logger()

# ==============================================================================
# 1. 🟢 自定义输出类：确保 mtp_loss 可以被传递
# ==============================================================================

def convert_router_logits_to_dict(router_logits) -> Optional[Dict[str, torch.Tensor]]:
    """
    将 router_logits 从 tuple 转换为 dict
    
    Args:
        router_logits: tuple of tensors 或 dict
    
    Returns:
        dict of tensors
    """
    if router_logits is None:
        return None
    
    if isinstance(router_logits, dict):
        return router_logits
    
    if isinstance(router_logits, (tuple, list)):
        return {f"layer_{i}": logits for i, logits in enumerate(router_logits)}
    
    return {"layer_0": router_logits}


def convert_tokens_per_expert_to_tensor(tokens) -> Optional[torch.Tensor]:
    """
    将 tokens_per_expert_global 转换为 Tensor
    
    Args:
        tokens: list, tuple, tensor, 或 None
    
    Returns:
        torch.Tensor 或 None
    """
    if tokens is None:
        return None
    
    if isinstance(tokens, torch.Tensor):
        return tokens
    
    if isinstance(tokens, (list, tuple)):
        if len(tokens) == 0:
            return torch.zeros(1, dtype=torch.long)
        return torch.tensor(tokens, dtype=torch.long)
    
    return torch.zeros(1, dtype=torch.long)


class MoEModelMTPOutputs(BaseModel):
    """MoE 模型 + MTP 的输出"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # 字段定义
    loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    mtp_loss: Optional[torch.Tensor] = None
    mtp_logits: Optional[List[torch.Tensor]] = None
    
    router_logits: Optional[Dict[str, torch.Tensor]] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Any] = None
    attentions: Optional[Any] = None
    aux_loss: Optional[torch.Tensor] = None
    
    def free_nongrad_feature(self): 
        return 
        
        
        # 保留以下字段（需要梯度）：
        # - loss: 需要用于反向传播
        # - mtp_loss: 需要用于反向传播
        # - aux_loss: 需要用于反向传播


# ==============================================================================
# 2. XTuner 配置类定义
# ==============================================================================
class Qwen3MoeMTPXTunerConfig(XTunerBaseModelConfig):
    """
    XTuner Config 包装器。
    用于在 SFT 配置文件中指定 HF Checkpoint 路径。
    """
    model_path: str | Path

    ep_size: int = 1
    float8_cfg: Optional[Any] = None
    model_config = ConfigDict(extra="allow")
    
    def build(self):
        # 触发下方包装类实例化
        return Qwen3MoeMTPXTunerModel(self)
        
    @property
    def hf_config(self):
        # 包装类不直接生成 HF Config，由内部的 HF 模型处理
        return None


# ==============================================================================
# 3. XTuner 模型包装类定义
# ==============================================================================
class Qwen3MoeMTPXTunerModel(XTunerBaseModel):
    """
    XTuner 模型包装器。
    内部封装了一个基于 trust_remote_code 加载的 Qwen3MoeForCausalLMMTP 实例。
    """
    def __init__(self, config: Qwen3MoeMTPXTunerConfig):
        super().__init__(config)
        
        logger.info(f"正在通过 XTuner 包装器加载模型: {config.model_path}")
        
        # 🟢 关键：直接通过 HF 接口加载。
        # 由于你的 Checkpoint 里已经注册了 auto_map，
        # 这里会自动实例化你手写的 Qwen3MoeForCausalLMMTP 类。
        with torch.device("cuda"):
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16, # 建议固定为 bf16
                device_map=None             # XTuner 内部会接管设备分配（FSDP/TP）
            )
        
        device = torch.cuda.current_device()

        if self.model.config.tie_word_embeddings:
            self.model.config.tie_word_embeddings = False
            old_lm_head = self.model.lm_head
            self.model.lm_head = nn.Linear(old_lm_head.in_features, old_lm_head.out_features, bias=False)
            self.model.lm_head.weight.data = old_lm_head.weight.data.clone()

        logger.info("Qwen3 MTP 模型加载并实例化完成！")

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "_no_split_modules"):
            if "MTPBlock" not in self.model._no_split_modules:
                self.model._no_split_modules.append("MTPBlock")
                logger.info("已将 MTPBlock 注册至 FSDP _no_split_modules")
        else:
            self.model._no_split_modules = ["Qwen3MoeDecoderLayer", "MTPBlock"]

        if hasattr(config, "float8_cfg") and config.float8_cfg is not None:
            if prepare_float8_for_fsdp is None:
                logger.warning("当前XTuner环境不支持float8")
            else:
                logger.info("正在应用Float8线性层转换")
                prepare_float8_for_fsdp(self.model, float8_cfg=config.float8_cfg)
                logger.info("模型已成功转换为FP8混合精度格式")

    def to_hf_key_list(self, key: str) -> list[str]:
        if "mtp_blocks" in key:
            if key.startswith("model."):
                return [key.replace("model.", "")]
            return [key]
        return [key]

    def from_hf(self, hf_path: str | Path, strict: bool = True) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        重写此方法。
        因为我们在 __init__ 里已经手动加载过权重了，所以这里返回空，
        防止 XTuner 尝试进行二次加载导致 Key 匹配失败。
        """
        return set(), set(), set()

    def save_hf(
        self,
        hf_dir: Path | str,
        save_dtype: torch.dtype = torch.bfloat16,
        safetensors_prefix: str = "model",
    ):
        """
        XTuner 触发保存时的逻辑。
        使用状态聚合（Gather）后通过 HF 原生方法保存。
        """
        with profile_time_and_memory(f"Saving MTP model to {hf_dir}"):
            hf_dir = Path(hf_dir)
            weight_map_dict: dict = {}
            
            # 1. 保存基础文件（tokenizer, configs, code files）
            # 我们通过内部模型的引用来完成
            self.model.config.save_pretrained(str(hf_dir))
            
            # 2. 收集全量 state_dict (支持 FSDP)
            state_dict = self._collect_full_state_dict(self.model)
            
            # 3. 仅 Rank 0 执行物理保存
            if dist.get_rank() == 0:
                # 注意：这里需要确保 key 名与加载时一致。
                # 理论上直接保存 state_dict 即可，HF 会处理分块。
                self.model.save_pretrained(
                    str(hf_dir),
                    state_dict=state_dict,
                    safe_serialization=True
                )
                
                # 更新索引文件
                update_weight_map_from_safetensors_index(weight_map_dict, hf_dir)
                with open(hf_dir / "model.safetensors.index.json", "w") as f:
                    json.dump({"weight_map": weight_map_dict, "metadata": {}}, f, indent=2)
            
            dist.barrier()
        return (set(self.state_dict()), set(), set())

    def forward(self, seq_ctx: Any, loss_ctx: Any) -> MoEModelMTPOutputs:
        """
        前向传播桥接。
        """
        # 1. 数据解包
        input_ids = seq_ctx.input_ids
        
        # 2. 获取标签 (尝试从 XTuner 不同的上下文位置提取)
        labels = None
        if hasattr(loss_ctx, "loss_kwargs") and hasattr(loss_ctx.loss_kwargs, "shifted_labels"):
            labels = loss_ctx.loss_kwargs.shifted_labels
        elif hasattr(seq_ctx, "labels"):
            labels = seq_ctx.labels
        elif isinstance(loss_ctx, dict) and "labels" in loss_ctx:
            labels = loss_ctx["labels"]
            
        if labels is None:
            # 兜底逻辑：如果 labels 彻底丢失，复制 input_ids
            labels = input_ids.clone()
            
        # 3. 调用带 MTP 逻辑的 HF 模型
        # 注意：kwargs 中的参数会传递给 Qwen3MoeForCausalLMMTP.forward
        hf_outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=getattr(seq_ctx, "attention_mask", None),
            position_ids=getattr(seq_ctx, "position_ids", None),
            output_router_logits=True, # 确保计算 MoE Aux Loss
            return_dict=True
        )
        
        # 4. 封装并传出 mtp_loss
        # return MoEModelMTPOutputs(
        #     loss=hf_outputs.loss,
        #     mtp_loss=getattr(hf_outputs, "mtp_loss", None), # 从 HF 输出中提取
        #     logits=hf_outputs.logits,
        #     hidden_states=hf_outputs.hidden_states,
        #     router_logits=hf_outputs.router_logits,
        # )

        # 转换 router_logits: tuple -> dict
        router_logits = convert_router_logits_to_dict(
            getattr(hf_outputs, 'router_logits', None)
        )
        
        # 转换 tokens_per_expert_global: list/tuple -> Tensor
        tokens_per_expert_global = convert_tokens_per_expert_to_tensor(
            getattr(hf_outputs, 'tokens_per_expert_global', None)
        )
        
        # ============ 返回正确格式的输出 ============
        return MoEModelMTPOutputs(
            # 主要输出
            loss=hf_outputs.loss,
            lm_loss=hf_outputs.lm_loss,
            logits=hf_outputs.logits,
            
            # ✅ 使用正确转换的格式
            router_logits=router_logits,  # Dict，不是 Tuple
            tokens_per_expert_global=tokens_per_expert_global,  # Tensor，不是 List
            
            # MTP 相关
            mtp_loss=getattr(hf_outputs, 'mtp_loss', None),
            mtp_logits=getattr(hf_outputs, 'mtp_logits', None),
            
            # 其他字段
            past_key_values=getattr(hf_outputs, 'past_key_values', None),
            hidden_states=getattr(hf_outputs, 'hidden_states', None),
            attentions=getattr(hf_outputs, 'attentions', None),
            aux_loss=getattr(hf_outputs, 'aux_loss', None),
        )
