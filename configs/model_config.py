"""
Model configuration classes for LISA-改.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class VisionEncoderConfig:
    """Configuration for Qwen2.5-VL Vision Encoder."""
    
    image_size: int = 896
    patch_size: int = 14
    hidden_dim: int = 1280
    num_layers: int = 32
    num_heads: int = 20
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    local_attn_window_size: int = 8  # 8x8 patches window
    global_attn_layers: List[int] = field(default_factory=lambda: [7, 15, 23, 31])
    hook_layers: List[int] = field(default_factory=lambda: [7, 15, 23, 31])
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    dropout_rate: float = 0.0
    
    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2
    
    @property
    def grid_size(self) -> Tuple[int, int]:
        size = self.image_size // self.patch_size
        return (size, size)


@dataclass
class AdapterConfig:
    """Configuration for Vision-to-SAM Adapter."""
    
    input_dim: int = 1280
    output_dim: int = 512
    use_layer_norm: bool = True
    drop_path_rate: float = 0.1


@dataclass
class SAMDecoderConfig:
    """Configuration for SAM 2.1 Mask Decoder."""
    
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 2
    num_mask_tokens: int = 4  # K masks
    iou_prediction_head_depth: int = 3
    iou_prediction_head_hidden_dim: int = 256
    mask_upscale_layers: int = 2  # 64x64 -> 256x256
    use_high_res_features: bool = True
    final_mask_size: int = 256


@dataclass
class DepthHeadConfig:
    """Configuration for Depth Estimation Head."""
    
    input_dim: int = 512
    hidden_dims: List[int] = field(default_factory=lambda: [256, 64])
    output_dim: int = 1
    output_size: int = 256
    use_gelu: bool = True
    
    # Depth extension flags for future v1
    enable: bool = False  # Set to True in v1 to enable depth estimation
    encoder_type: str = "depth_anything_v2"  # Future: depth_anything_v2, midas, etc.
    use_depth_tokens: bool = False  # Future: inject depth features into LLM


@dataclass
class LLMIntegrationConfig:
    """Configuration for LLM Integration."""
    
    llm_hidden_dim: int = 2048
    mask_token_proj_dim: int = 512
    special_tokens: List[str] = field(default_factory=lambda: ["<SEG>", "<VIS_SUM>", "<DEPTH>", "<DEPTH_SUM>"])
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    """Configuration for Training."""
    
    # Optimizer
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Scheduler
    warmup_steps: int = 500
    lr_scheduler_type: str = "cosine"
    
    # Training settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    num_epochs: int = 10
    
    # Loss weights
    text_loss_weight: float = 1.0
    mask_loss_weight: float = 1.0
    iou_loss_weight: float = 0.1
    depth_loss_weight: float = 0.3
    
    # Mixed precision
    use_fp16: bool = True
    use_deepspeed: bool = False
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Data
    train_data_paths: List[str] = field(default_factory=list)
    val_data_paths: List[str] = field(default_factory=list)


@dataclass
class LISAConfig:
    """Main configuration for LISA-改 model."""
    
    model_name: str = "lisa-qwen2.5-sam2.1"
    model_size: str = "3B"  # 3B, 7B, or 72B
    
    vision_encoder: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    sam_decoder: SAMDecoderConfig = field(default_factory=SAMDecoderConfig)
    depth_head: Optional[DepthHeadConfig] = field(default_factory=DepthHeadConfig)
    llm_integration: LLMIntegrationConfig = field(default_factory=LLMIntegrationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Model paths
    qwen_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    sam_model_path: str = "facebook/sam2.1-hiera-large"
    
    # Device settings
    device: str = "cuda"
    dtype: str = "float16"  # float16, bfloat16, or float32