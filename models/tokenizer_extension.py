"""
Tokenizer extension module for LISA-改.

This module extends the Qwen2.5-VL tokenizer with special tokens
for segmentation tasks.
"""

from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class ExtendedQwenTokenizer:
    """Extended Qwen2.5-VL tokenizer with special tokens for LISA-改."""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        special_tokens: Optional[List[str]] = None,
        device: str = "cuda",
    ):
        """
        Initialize the extended tokenizer.
        
        Args:
            model_path: Path to the Qwen model or HuggingFace model ID
            special_tokens: List of special tokens to add. Defaults to ["<SEG>", "<VIS_SUM>"]
            device: Device to use for embeddings extension
        """
        self.device = device
        self.model_path = model_path
        
        # Load base tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Define special tokens
        if special_tokens is None:
            special_tokens = ["<SEG>", "<VIS_SUM>", "<MASK>", "<DEPTH>"]
        
        self.special_tokens = special_tokens
        self.special_token_ids = {}
        
        # Add special tokens
        self._add_special_tokens()
        
        logger.info(f"Tokenizer initialized with {len(self.tokenizer)} tokens")
        logger.info(f"Added special tokens: {self.special_tokens}")
    
    def _add_special_tokens(self) -> None:
        """Add special tokens to the tokenizer."""
        # Check which tokens need to be added
        tokens_to_add = []
        for token in self.special_tokens:
            if token not in self.tokenizer.get_vocab():
                tokens_to_add.append(token)
        
        if tokens_to_add:
            # Add new tokens
            num_added = self.tokenizer.add_tokens(tokens_to_add, special_tokens=True)
            logger.info(f"Added {num_added} new tokens to tokenizer")
            
            # Update special tokens map
            for token in tokens_to_add:
                # Add to additional_special_tokens if not already there
                if hasattr(self.tokenizer, 'additional_special_tokens'):
                    if token not in self.tokenizer.additional_special_tokens:
                        self.tokenizer.add_special_tokens({
                            'additional_special_tokens': 
                            self.tokenizer.additional_special_tokens + [token]
                        })
                else:
                    self.tokenizer.add_special_tokens({
                        'additional_special_tokens': [token]
                    })
        
        # Store token IDs for quick access
        for token in self.special_tokens:
            self.special_token_ids[token] = self.tokenizer.convert_tokens_to_ids(token)
            logger.info(f"Token '{token}' has ID: {self.special_token_ids[token]}")
    
    def get_seg_token_id(self) -> int:
        """Get the token ID for <SEG>."""
        return self.special_token_ids.get("<SEG>", -1)
    
    def get_vis_sum_token_id(self) -> int:
        """Get the token ID for <VIS_SUM>."""
        return self.special_token_ids.get("<VIS_SUM>", -1)
    
    def get_mask_token_id(self) -> int:
        """Get the token ID for <MASK>."""
        return self.special_token_ids.get("<MASK>", -1)
    
    def get_depth_token_id(self) -> int:
        """Get the token ID for <DEPTH>."""
        return self.special_token_ids.get("<DEPTH>", -1)
    
    def resize_model_embeddings(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Resize model embeddings to accommodate new tokens.
        
        Args:
            model: The model whose embeddings need to be resized
            
        Returns:
            The model with resized embeddings
        """
        original_vocab_size = model.get_input_embeddings().weight.shape[0]
        new_vocab_size = len(self.tokenizer)
        
        if new_vocab_size > original_vocab_size:
            model.resize_token_embeddings(new_vocab_size)
            logger.info(
                f"Resized model embeddings from {original_vocab_size} to {new_vocab_size}"
            )
            
            # Initialize new embeddings with random values
            # This is a simple initialization; you might want to use
            # more sophisticated methods in practice
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
            
            # Initialize new embeddings as average of existing ones
            # for better starting point
            avg_input_embedding = input_embeddings[:original_vocab_size].mean(dim=0)
            avg_output_embedding = output_embeddings[:original_vocab_size].mean(dim=0)
            
            for i in range(original_vocab_size, new_vocab_size):
                # Add small random noise to break symmetry
                noise = torch.randn_like(avg_input_embedding) * 0.01
                input_embeddings[i] = avg_input_embedding + noise
                output_embeddings[i] = avg_output_embedding + noise
            
            logger.info("Initialized new token embeddings")
        
        return model
    
    def prepare_text_for_seg(self, text: str, add_seg_token: bool = True) -> str:
        """
        Prepare text for segmentation by optionally adding <SEG> token.
        
        Args:
            text: Input text
            add_seg_token: Whether to add <SEG> token at the end
            
        Returns:
            Prepared text
        """
        if add_seg_token and not text.endswith("<SEG>"):
            text = text.strip() + " <SEG>"
        return text
    
    def encode_with_special_tokens(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = "pt",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Encode text while preserving special tokens.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            return_tensors: Return type ("pt" for PyTorch tensors)
            **kwargs: Additional arguments for tokenizer
            
        Returns:
            Encoded inputs
        """
        # Ensure special tokens are not split
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            **kwargs
        )
        
        return encoded
    
    def decode_skip_special(self, token_ids: torch.Tensor, skip_seg: bool = True) -> str:
        """
        Decode tokens while optionally skipping <SEG> tokens.
        
        Args:
            token_ids: Token IDs to decode
            skip_seg: Whether to skip <SEG> tokens in output
            
        Returns:
            Decoded text
        """
        if skip_seg:
            # Filter out <SEG> tokens
            seg_id = self.get_seg_token_id()
            if seg_id != -1:
                token_ids = token_ids[token_ids != seg_id]
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
    
    def save_tokenizer(self, save_path: str) -> None:
        """Save the extended tokenizer."""
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Tokenizer saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "ExtendedQwenTokenizer":
        """Load a previously saved extended tokenizer."""
        instance = cls(model_path=path, **kwargs)
        return instance
    
    def __call__(self, *args, **kwargs):
        """Forward calls to the underlying tokenizer."""
        return self.tokenizer(*args, **kwargs)
    
    def __getattr__(self, name):
        """Forward attribute access to the underlying tokenizer."""
        return getattr(self.tokenizer, name)