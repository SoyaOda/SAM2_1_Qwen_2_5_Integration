"""
Minimal test for basic functionality.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.model_config import LISAConfig

def test_minimal():
    """Test minimal configuration loading."""
    print("Testing minimal configuration...")
    
    # Create config
    config = LISAConfig(model_size="3B")
    print(f"Model size: {config.model_size}")
    print(f"Device: {config.device}")
    print(f"Vision encoder image size: {config.vision_encoder.image_size}")
    print(f"SAM decoder transformer dim: {config.sam_decoder.transformer_dim}")
    
    # Test tokenizer
    print("\nTesting tokenizer...")
    from models.tokenizer_extension import ExtendedQwenTokenizer
    tokenizer = ExtendedQwenTokenizer(
        model_path=config.qwen_model_path,
        device="cpu"
    )
    print(f"Tokenizer vocabulary size: {len(tokenizer.tokenizer)}")
    print(f"SEG token ID: {tokenizer.get_seg_token_id()}")
    
    # Test simple text
    text = "This is a test <SEG>"
    tokens = tokenizer(text)
    print(f"Test text: '{text}'")
    print(f"Token IDs: {tokens['input_ids']}")
    
    print("\n✓ Minimal test passed!")

if __name__ == "__main__":
    test_minimal()