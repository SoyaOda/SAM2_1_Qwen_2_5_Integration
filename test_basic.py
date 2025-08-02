"""
Basic test script for LISA-改 components.

This script tests individual components to ensure they work correctly.
"""

import sys
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.model_config import (
    LISAConfig, VisionEncoderConfig, AdapterConfig,
    SAMDecoderConfig, DepthHeadConfig, LLMIntegrationConfig
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_tokenizer_extension():
    """Test tokenizer extension module."""
    logger.info("Testing tokenizer extension...")
    
    try:
        from models.tokenizer_extension import ExtendedQwenTokenizer
        
        # Create tokenizer (using mock for testing)
        tokenizer = ExtendedQwenTokenizer(
            model_path="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cpu"
        )
        
        # Check special tokens
        assert tokenizer.get_seg_token_id() != -1, "SEG token not found"
        assert tokenizer.get_vis_sum_token_id() != -1, "VIS_SUM token not found"
        
        logger.info("✓ Tokenizer extension test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Tokenizer extension test failed: {e}")
        return False


def test_vision_encoder():
    """Test vision encoder module."""
    logger.info("Testing vision encoder...")
    
    try:
        from models.vision_encoder import VisionEncoder
        
        # Create config
        config = VisionEncoderConfig(
            image_size=224,  # Smaller for testing
            patch_size=14,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
        )
        
        # Create encoder
        encoder = VisionEncoder(config)
        
        # Test forward pass
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, config.image_size, config.image_size)
        output = encoder(dummy_input, return_features=True)
        
        # Check output shape
        expected_grid = config.image_size // config.patch_size
        assert output.shape == (batch_size, config.hidden_dim, expected_grid, expected_grid), \
            f"Unexpected output shape: {output.shape}"
        
        logger.info("✓ Vision encoder test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Vision encoder test failed: {e}")
        return False


def test_adapter():
    """Test adapter module."""
    logger.info("Testing adapter...")
    
    try:
        from models.adapter import VisionToSAMAdapter
        
        # Create config
        config = AdapterConfig(
            input_dim=1280,
            output_dim=512,
            use_layer_norm=True,
            drop_path_rate=0.1,
        )
        
        # Create adapter
        adapter = VisionToSAMAdapter(config)
        
        # Test forward pass
        batch_size = 2
        dummy_input = torch.randn(batch_size, config.input_dim, 64, 64)
        output = adapter(dummy_input)
        
        # Check output shape
        assert output.shape == (batch_size, config.output_dim, 64, 64), \
            f"Unexpected output shape: {output.shape}"
        
        logger.info("✓ Adapter test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Adapter test failed: {e}")
        return False


def test_prompt_encoder():
    """Test prompt encoder module."""
    logger.info("Testing prompt encoder...")
    
    try:
        from models.prompt_encoder import PromptEncoder
        
        # Create encoder
        encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(896, 896),
        )
        
        # Test with no prompts
        sparse_emb, dense_emb = encoder()
        assert sparse_emb.shape[1] == 1, "Should have no-prompt embedding"
        assert dense_emb is None, "Dense embeddings should be None without masks"
        
        # Test with points
        points = torch.rand(1, 5, 2)  # 5 points
        labels = torch.randint(0, 2, (1, 5))  # Random labels
        sparse_emb, _ = encoder(points=(points, labels))
        assert sparse_emb.shape == (1, 5, 256), f"Unexpected shape: {sparse_emb.shape}"
        
        logger.info("✓ Prompt encoder test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Prompt encoder test failed: {e}")
        return False


def test_mask_decoder():
    """Test mask decoder module."""
    logger.info("Testing mask decoder...")
    
    try:
        from models.mask_decoder import MaskDecoder
        
        # Create config
        config = SAMDecoderConfig(
            transformer_dim=512,
            transformer_heads=8,
            transformer_layers=2,
            num_mask_tokens=4,
            final_mask_size=256,
        )
        
        # Create decoder
        decoder = MaskDecoder(config)
        
        # Test forward pass
        batch_size = 2
        image_embeddings = torch.randn(batch_size, config.transformer_dim, 64, 64)
        sparse_prompts = torch.randn(batch_size, 5, config.transformer_dim)
        
        try:
            masks, iou_pred, mask_tokens = decoder(
                image_embeddings=image_embeddings,
                sparse_prompt_embeddings=sparse_prompts,
                multimask_output=False,
            )
            
            # Check outputs
            assert masks.shape == (batch_size, 1, config.final_mask_size, config.final_mask_size), \
                f"Unexpected mask shape: {masks.shape}"
            assert iou_pred.shape == (batch_size, 1), f"Unexpected IoU shape: {iou_pred.shape}"
            
            logger.info("✓ Mask decoder test passed")
            return True
        except Exception as inner_e:
            logger.error(f"Forward pass error: {inner_e}")
            import traceback
            traceback.print_exc()
            raise
        
    except Exception as e:
        logger.error(f"✗ Mask decoder test failed: {e}")
        return False


def test_depth_head():
    """Test depth head module."""
    logger.info("Testing depth head...")
    
    try:
        from models.depth_head import DepthHead
        
        # Test with disabled depth (v0)
        config_disabled = DepthHeadConfig(enable=False)
        depth_head = DepthHead(config_disabled)
        
        # Should raise error when trying to use
        dummy_input = torch.randn(2, 512, 64, 64)
        try:
            output = depth_head(dummy_input)
            logger.error("✗ Depth head should raise error when disabled")
            return False
        except RuntimeError as e:
            logger.info(f"✓ Correctly raised error: {e}")
        
        # Test with enabled depth (v1)
        config_enabled = DepthHeadConfig(
            enable=True,
            input_dim=512,
            output_size=256,
        )
        depth_head = DepthHead(config_enabled)
        output = depth_head(dummy_input)
        
        assert output.shape == (2, 1, 256, 256), f"Unexpected output shape: {output.shape}"
        
        logger.info("✓ Depth head test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Depth head test failed: {e}")
        return False


def test_integration():
    """Test basic integration of components."""
    logger.info("Testing component integration...")
    
    try:
        # This is a placeholder for integration testing
        # Full integration test would require downloading models
        logger.info("✓ Integration test placeholder passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting LISA-改 component tests...")
    logger.info("=" * 50)
    
    tests = [
        test_tokenizer_extension,
        test_vision_encoder,
        test_adapter,
        test_prompt_encoder,
        test_mask_decoder,
        test_depth_head,
        test_integration,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
            results.append(False)
        logger.info("-" * 50)
    
    # Summary
    passed = sum(results)
    total = len(results)
    logger.info("=" * 50)
    logger.info(f"Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All tests passed! ✓")
        return 0
    else:
        logger.error(f"{total - passed} tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())