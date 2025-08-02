"""
Vision Encoder V2 for LISA-改.

This module implements a more accurate vision encoder based on
Qwen2.5-VL's official implementation, using the pretrained model
directly rather than reimplementing from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import logging
import math

from utils.model_utils import load_pretrained_qwen_vl, extract_vision_features_from_qwen
from configs.model_config import VisionEncoderConfig

logger = logging.getLogger(__name__)


class VisionEncoderV2(nn.Module):
    """
    Vision Encoder V2 using pretrained Qwen2.5-VL.
    
    This implementation directly uses the official Qwen2.5-VL vision model
    to ensure compatibility and correctness.
    """
    
    def __init__(
        self,
        config: VisionEncoderConfig,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        use_pretrained: bool = True,
    ):
        super().__init__()
        self.config = config
        self.model_name = model_name
        
        if use_pretrained:
            # Load pretrained Qwen2.5-VL
            try:
                qwen_model, self.processor = load_pretrained_qwen_vl(
                    model_name=model_name,
                    device="cpu",  # Will move to device later
                    torch_dtype=torch.float32,  # Load in full precision first
                )
                
                # Extract vision model
                self.vision_model = qwen_model.visual
                
                # Update config based on loaded model
                if hasattr(self.vision_model, "config"):
                    model_config = self.vision_model.config
                    self.hidden_dim = model_config.hidden_size
                    self.patch_size = model_config.patch_size
                    self.num_heads = model_config.num_attention_heads
                else:
                    # Fallback to config values
                    self.hidden_dim = config.hidden_dim
                    self.patch_size = config.patch_size
                    self.num_heads = config.num_heads
                
                logger.info(f"Loaded pretrained vision encoder from {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load pretrained model: {e}")
                raise RuntimeError(
                    f"Cannot load pretrained Qwen2.5-VL. "
                    f"Please install: pip install transformers>=4.37.0\n"
                    f"Error: {e}"
                )
        else:
            raise NotImplementedError(
                "Training from scratch not supported in V2. "
                "Please use pretrained=True or use VisionEncoder V1."
            )
        
        # Hook storage for intermediate features
        self._intermediate_features = {}
        self._hooks = []
        
        # Register hooks if needed
        if config.hook_layers:
            self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        # Clear existing hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._intermediate_features.clear()
        
        # Register new hooks
        if hasattr(self.vision_model, "blocks"):
            # For transformer-based models
            for idx in self.config.hook_layers:
                if idx < len(self.vision_model.blocks):
                    hook = self.vision_model.blocks[idx].register_forward_hook(
                        self._make_hook(f"block_{idx}")
                    )
                    self._hooks.append(hook)
                else:
                    logger.warning(f"Hook layer {idx} exceeds model depth")
        else:
            logger.warning("Cannot register hooks: model structure not recognized")
    
    def _make_hook(self, name: str):
        """Create a hook function."""
        def hook(module, input, output):
            self._intermediate_features[name] = output
        return hook
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through vision encoder.
        
        Args:
            pixel_values: Input images (B, C, H, W) or (B, T, C, H, W) for video
            image_grid_thw: Grid dimensions [temporal, height, width]
            return_dict: Whether to return dictionary output
            
        Returns:
            Dictionary containing:
                - last_hidden_state: Final features (B, N, D)
                - feature_map: Reshaped 2D features (B, D, H', W')
                - intermediate_features: Dict of hooked features
        """
        # Input validation
        if pixel_values.dim() not in [4, 5]:
            raise ValueError(
                f"Expected 4D (image) or 5D (video) input, got {pixel_values.dim()}D"
            )
        
        # Process through vision model
        try:
            outputs = self.vision_model(
                pixel_values=pixel_values,
                grid_thw=image_grid_thw,
            )
            
            # Extract features
            hidden_states = outputs.last_hidden_state  # (B, N, D)
            
            # Calculate spatial dimensions
            B, N, D = hidden_states.shape
            # Assuming square spatial arrangement
            H = W = int(math.sqrt(N))
            
            if H * W != N:
                # Try to find the closest factors
                factors = []
                for i in range(1, int(math.sqrt(N)) + 1):
                    if N % i == 0:
                        factors.append((i, N // i))
                
                if factors:
                    # Choose the most square-like factors
                    H, W = min(factors, key=lambda x: abs(x[0] - x[1]))
                else:
                    # Fallback: pad to nearest square
                    H = W = int(math.ceil(math.sqrt(N)))
                    if H * W > N:
                        # Pad hidden states
                        pad_size = H * W - N
                        padding = torch.zeros(
                            B, pad_size, D,
                            dtype=hidden_states.dtype,
                            device=hidden_states.device
                        )
                        hidden_states = torch.cat([hidden_states, padding], dim=1)
            
            # Reshape to 2D feature map
            feature_map = hidden_states.reshape(B, H, W, D).permute(0, 3, 1, 2)
            
            if return_dict:
                return {
                    "last_hidden_state": hidden_states[:, :N],  # Remove padding if any
                    "feature_map": feature_map,
                    "intermediate_features": self._intermediate_features.copy(),
                    "grid_size": (H, W),
                }
            else:
                return feature_map
                
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise RuntimeError(
                f"Vision encoder forward pass failed. "
                f"Input shape: {pixel_values.shape}, "
                f"Error: {e}"
            )
    
    def get_output_dim(self) -> int:
        """Get output dimension of vision encoder."""
        return self.hidden_dim
    
    def get_patch_size(self) -> int:
        """Get patch size."""
        return self.patch_size
    
    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Vision encoder frozen")
    
    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Vision encoder unfrozen")
    
    def __del__(self):
        """Cleanup hooks when object is deleted."""
        for hook in self._hooks:
            hook.remove()


class VisionEncoderWrapper(nn.Module):
    """
    Wrapper to make VisionEncoderV2 compatible with the original interface.
    """
    
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        
        # Determine model name based on size
        model_map = {
            "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
            "7B": "Qwen/Qwen2.5-VL-7B-Instruct",
            "72B": "Qwen/Qwen2.5-VL-72B-Instruct",
        }
        
        model_size = getattr(config, "model_size", "3B")
        model_name = model_map.get(model_size, "Qwen/Qwen2.5-VL-3B-Instruct")
        
        # Create V2 encoder
        self.encoder = VisionEncoderV2(
            config=config,
            model_name=model_name,
            use_pretrained=True,
        )
        
        # Update config with actual values
        config.hidden_dim = self.encoder.get_output_dim()
        config.patch_size = self.encoder.get_patch_size()
    
    def forward(self, x: torch.Tensor, return_features: bool = True) -> torch.Tensor:
        """
        Forward pass with original interface.
        
        Args:
            x: Input images (B, 3, H, W)
            return_features: Whether to return 2D feature map
            
        Returns:
            Feature map (B, C, H', W') or sequence (B, N, C)
        """
        outputs = self.encoder(x, return_dict=True)
        
        if return_features:
            return outputs["feature_map"]
        else:
            return outputs["last_hidden_state"]
    
    def register_hooks(self):
        """Register hooks (compatibility method)."""
        # Already handled in V2 init
        pass
    
    def get_intermediate_features(self) -> Dict[str, torch.Tensor]:
        """Get intermediate features."""
        return self.encoder._intermediate_features.copy()