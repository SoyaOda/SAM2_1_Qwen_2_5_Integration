"""
Model utilities for LISA-改.

This module provides utility functions for model initialization,
pretrained weight loading, and architecture-specific operations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
import logging
import warnings
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


def load_pretrained_qwen_vl(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> Tuple[Any, Any]:
    """
    Load pretrained Qwen2.5-VL model and processor.
    
    Args:
        model_name: HuggingFace model ID
        device: Device to load model on
        torch_dtype: Data type for model weights
        
    Returns:
        model: Pretrained Qwen2.5-VL model
        processor: Associated processor
    """
    try:
        # Load model with proper configuration
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
            # Enable flash attention if available
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        logger.info(f"Successfully loaded Qwen2.5-VL model: {model_name}")
        return model, processor
        
    except Exception as e:
        logger.error(f"Failed to load Qwen2.5-VL model: {e}")
        raise RuntimeError(f"Cannot load pretrained model {model_name}: {e}")


def load_pretrained_sam2(
    checkpoint_path: str = "facebook/sam2.1-hiera-large",
    device: str = "cuda",
) -> Any:
    """
    Load pretrained SAM 2.1 model.
    
    Args:
        checkpoint_path: Path to checkpoint or HuggingFace model ID
        device: Device to load model on
        
    Returns:
        SAM2 model instance
    """
    try:
        # Try loading from HuggingFace first
        if "/" in checkpoint_path and not checkpoint_path.startswith("/"):
            # Looks like a HuggingFace model ID
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
            ).to(device)
        else:
            # Load from local checkpoint
            # This would require SAM2 package installation
            raise NotImplementedError(
                "Local SAM2 checkpoint loading not implemented. "
                "Please use HuggingFace model ID or install SAM2 package."
            )
        
        logger.info(f"Successfully loaded SAM2 model: {checkpoint_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load SAM2 model: {e}")
        raise RuntimeError(f"Cannot load SAM2 model {checkpoint_path}: {e}")


def extract_vision_features_from_qwen(
    qwen_model: Any,
    pixel_values: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract vision features from Qwen2.5-VL model.
    
    Args:
        qwen_model: Qwen2.5-VL model instance
        pixel_values: Input images (B, C, H, W)
        image_grid_thw: Grid dimensions (temporal, height, width)
        
    Returns:
        Dictionary containing vision features
    """
    try:
        # Get vision model from Qwen
        vision_model = qwen_model.visual
        
        # Process through vision encoder
        vision_outputs = vision_model(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
        )
        
        # Extract features
        features = {
            "last_hidden_state": vision_outputs.last_hidden_state,
            "hidden_states": vision_outputs.hidden_states if hasattr(vision_outputs, "hidden_states") else None,
        }
        
        return features
        
    except AttributeError as e:
        logger.error(f"Model structure mismatch: {e}")
        raise RuntimeError(
            "Cannot extract vision features. Make sure you're using Qwen2.5-VL model."
        )


def create_vision_attention_mask(
    batch_size: int,
    seq_length: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create attention mask for vision transformer.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        dtype: Data type
        device: Device
        
    Returns:
        Attention mask
    """
    return torch.ones(
        (batch_size, seq_length),
        dtype=dtype,
        device=device,
    )


def compute_vision_position_ids(
    seq_length: int,
    grid_h: int,
    grid_w: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute position IDs for vision transformer grid.
    
    Args:
        seq_length: Total sequence length
        grid_h: Grid height
        grid_w: Grid width
        device: Device
        
    Returns:
        Position IDs tensor
    """
    # Create 2D position grid
    pos_h = torch.arange(grid_h, device=device)
    pos_w = torch.arange(grid_w, device=device)
    
    # Create mesh grid
    pos_h = pos_h.unsqueeze(1).expand(-1, grid_w)
    pos_w = pos_w.unsqueeze(0).expand(grid_h, -1)
    
    # Flatten and combine
    position_ids = pos_h.flatten() * grid_w + pos_w.flatten()
    
    return position_ids


def validate_model_compatibility(
    vision_model: Any,
    sam_model: Any,
) -> bool:
    """
    Validate compatibility between vision encoder and SAM decoder.
    
    Args:
        vision_model: Vision encoder model
        sam_model: SAM decoder model
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        # Check vision output dimension
        if hasattr(vision_model, "config"):
            vision_dim = getattr(vision_model.config, "hidden_size", None)
        else:
            vision_dim = None
            
        # Check SAM input dimension
        if hasattr(sam_model, "image_encoder"):
            sam_dim = getattr(sam_model.image_encoder, "embed_dim", None)
        else:
            sam_dim = None
            
        if vision_dim and sam_dim:
            if vision_dim != sam_dim:
                warnings.warn(
                    f"Dimension mismatch: Vision encoder outputs {vision_dim}, "
                    f"but SAM expects {sam_dim}. Adapter layer required."
                )
                return False
                
        return True
        
    except Exception as e:
        logger.warning(f"Cannot validate model compatibility: {e}")
        return True  # Assume compatible if cannot check