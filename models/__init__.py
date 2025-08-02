"""
Models package for LISA-改 implementation.

This package contains all model components for integrating 
Qwen2.5-VL and SAM 2.1.
"""

from .vision_encoder import VisionEncoder
from .adapter import VisionToSAMAdapter, PromptEmbeddingAdapter, MultiScaleFeatureAdapter
from .mask_decoder import MaskDecoder, TwoWayTransformer
from .prompt_encoder import PromptEncoder
from .depth_head import DepthHead, DepthTokenAdapter, compute_depth_loss
from .llm_integration import LLMIntegration, SegmentationTokenHandler
from .tokenizer_extension import ExtendedQwenTokenizer
from .lisa_model import LISAModel

__all__ = [
    # Main model
    "LISAModel",
    
    # Core components
    "VisionEncoder",
    "MaskDecoder",
    "PromptEncoder",
    "DepthHead",
    "LLMIntegration",
    
    # Adapters
    "VisionToSAMAdapter",
    "PromptEmbeddingAdapter",
    "MultiScaleFeatureAdapter",
    "DepthTokenAdapter",
    
    # Utilities
    "ExtendedQwenTokenizer",
    "TwoWayTransformer",
    "SegmentationTokenHandler",
    "compute_depth_loss",
]