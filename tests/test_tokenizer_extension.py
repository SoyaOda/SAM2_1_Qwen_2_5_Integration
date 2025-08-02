"""
Tests for tokenizer extension module.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.append("..")

from models.tokenizer_extension import ExtendedQwenTokenizer


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.vocab = {
            "hello": 0,
            "world": 1,
            "<|im_start|>": 2,
            "<|im_end|>": 3,
        }
        self.added_tokens = []
        self.additional_special_tokens = []
        
    def get_vocab(self):
        return self.vocab
    
    def add_tokens(self, tokens, special_tokens=True):
        count = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.added_tokens.append(token)
                count += 1
        return count
    
    def add_special_tokens(self, special_tokens_dict):
        if 'additional_special_tokens' in special_tokens_dict:
            self.additional_special_tokens.extend(
                special_tokens_dict['additional_special_tokens']
            )
    
    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, -1)
    
    def __len__(self):
        return len(self.vocab)
    
    def __call__(self, text, **kwargs):
        # Simple mock encoding
        return {"input_ids": torch.tensor([[0, 1]])}
    
    def decode(self, token_ids, skip_special_tokens=False):
        return "decoded text"
    
    def save_pretrained(self, path):
        pass


@patch('models.tokenizer_extension.AutoTokenizer')
class TestExtendedQwenTokenizer:
    """Test cases for ExtendedQwenTokenizer."""
    
    def test_initialization(self, mock_auto_tokenizer):
        """Test tokenizer initialization."""
        mock_auto_tokenizer.from_pretrained.return_value = MockTokenizer()
        
        tokenizer = ExtendedQwenTokenizer(device="cpu")
        
        # Check that special tokens were added
        assert "<SEG>" in tokenizer.special_tokens
        assert "<VIS_SUM>" in tokenizer.special_tokens
        assert "<MASK>" in tokenizer.special_tokens
        assert "<DEPTH>" in tokenizer.special_tokens
        
        # Check that token IDs were stored
        assert tokenizer.get_seg_token_id() != -1
        assert tokenizer.get_vis_sum_token_id() != -1
    
    def test_custom_special_tokens(self, mock_auto_tokenizer):
        """Test initialization with custom special tokens."""
        mock_auto_tokenizer.from_pretrained.return_value = MockTokenizer()
        
        custom_tokens = ["<CUSTOM1>", "<CUSTOM2>"]
        tokenizer = ExtendedQwenTokenizer(
            special_tokens=custom_tokens,
            device="cpu"
        )
        
        assert tokenizer.special_tokens == custom_tokens
        assert len(tokenizer.special_token_ids) == len(custom_tokens)
    
    def test_resize_model_embeddings(self, mock_auto_tokenizer):
        """Test model embeddings resizing."""
        mock_auto_tokenizer.from_pretrained.return_value = MockTokenizer()
        tokenizer = ExtendedQwenTokenizer(device="cpu")
        
        # Create mock model
        mock_model = Mock()
        mock_input_embeddings = Mock()
        mock_output_embeddings = Mock()
        
        # Set up embedding weights
        original_size = 4
        mock_input_embeddings.weight.shape = [original_size, 768]
        mock_output_embeddings.weight.shape = [original_size, 768]
        
        # Create actual tensor data for embeddings
        mock_input_embeddings.weight.data = torch.randn(8, 768)
        mock_output_embeddings.weight.data = torch.randn(8, 768)
        
        mock_model.get_input_embeddings.return_value = mock_input_embeddings
        mock_model.get_output_embeddings.return_value = mock_output_embeddings
        
        # Resize embeddings
        resized_model = tokenizer.resize_model_embeddings(mock_model)
        
        # Check that resize was called
        mock_model.resize_token_embeddings.assert_called_once()
    
    def test_prepare_text_for_seg(self, mock_auto_tokenizer):
        """Test text preparation for segmentation."""
        mock_auto_tokenizer.from_pretrained.return_value = MockTokenizer()
        tokenizer = ExtendedQwenTokenizer(device="cpu")
        
        # Test adding SEG token
        text = "Segment this object"
        prepared = tokenizer.prepare_text_for_seg(text, add_seg_token=True)
        assert prepared.endswith("<SEG>")
        
        # Test not adding SEG token
        prepared = tokenizer.prepare_text_for_seg(text, add_seg_token=False)
        assert not prepared.endswith("<SEG>")
        
        # Test when SEG token already present
        text_with_seg = "Segment this object <SEG>"
        prepared = tokenizer.prepare_text_for_seg(text_with_seg, add_seg_token=True)
        assert prepared.count("<SEG>") == 1
    
    def test_encode_with_special_tokens(self, mock_auto_tokenizer):
        """Test encoding with special tokens."""
        mock_tokenizer = MockTokenizer()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        tokenizer = ExtendedQwenTokenizer(device="cpu")
        
        text = "Hello world <SEG>"
        encoded = tokenizer.encode_with_special_tokens(text)
        
        assert "input_ids" in encoded
        assert isinstance(encoded["input_ids"], torch.Tensor)
    
    def test_decode_skip_special(self, mock_auto_tokenizer):
        """Test decoding with special token skipping."""
        mock_tokenizer = MockTokenizer()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        tokenizer = ExtendedQwenTokenizer(device="cpu")
        
        # Mock token IDs including SEG token
        seg_id = tokenizer.get_seg_token_id()
        token_ids = torch.tensor([0, 1, seg_id, 2])
        
        # Test with skipping SEG
        decoded = tokenizer.decode_skip_special(token_ids, skip_seg=True)
        assert decoded == "decoded text"
        
        # Test without skipping SEG
        decoded = tokenizer.decode_skip_special(token_ids, skip_seg=False)
        assert decoded == "decoded text"
    
    def test_special_token_getters(self, mock_auto_tokenizer):
        """Test special token ID getters."""
        mock_auto_tokenizer.from_pretrained.return_value = MockTokenizer()
        tokenizer = ExtendedQwenTokenizer(device="cpu")
        
        assert isinstance(tokenizer.get_seg_token_id(), int)
        assert isinstance(tokenizer.get_vis_sum_token_id(), int)
        assert isinstance(tokenizer.get_mask_token_id(), int)
        assert isinstance(tokenizer.get_depth_token_id(), int)
        
        # All should be valid IDs (not -1)
        assert tokenizer.get_seg_token_id() >= 0
        assert tokenizer.get_vis_sum_token_id() >= 0
    
    def test_save_and_load(self, mock_auto_tokenizer):
        """Test saving and loading tokenizer."""
        mock_tokenizer = MockTokenizer()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = ExtendedQwenTokenizer(device="cpu")
        
        # Test save
        save_path = "/tmp/test_tokenizer"
        tokenizer.save_tokenizer(save_path)
        
        # Test load
        loaded_tokenizer = ExtendedQwenTokenizer.from_pretrained(save_path)
        assert isinstance(loaded_tokenizer, ExtendedQwenTokenizer)