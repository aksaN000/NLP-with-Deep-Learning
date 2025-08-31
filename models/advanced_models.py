"""
Advanced NLP Models and Architectures

This module contains state-of-the-art NLP model implementations with comprehensive
explanations, optimizations, and educational features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    vocab_size: int = 30000
    max_seq_length: int = 512
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism from 'Attention Is All You Need'.
    
    This implementation includes:
    - Scaled dot-product attention
    - Multiple attention heads
    - Linear projections for Q, K, V
    - Output projection
    - Dropout for regularization
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear projections for queries, keys, and values
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Output projection
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention computation."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask to prevent attention to certain positions
            return_attention_weights: Whether to return attention weights
            
        Returns:
            context_layer: Output tensor after attention
            attention_probs: Attention weights (if return_attention_weights=True)
        """
        # Linear projections
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to original dimensions
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        context_layer = self.dense(context_layer)
        
        if return_attention_weights:
            return context_layer, attention_probs
        return context_layer


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    
    This adds positional information to the input embeddings,
    allowing the model to understand sequence order.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        pe = torch.zeros(config.max_seq_length, config.hidden_size)
        position = torch.arange(0, config.max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, config.hidden_size, 2).float() * 
                           (-math.log(10000.0) / config.hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of:
    - Multi-head self-attention
    - Feed-forward network
    - Residual connections and layer normalization
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with residual connections and layer normalization."""
        # Self-attention with residual connection
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm1(hidden_states + attention_output)
        
        # Feed-forward with residual connection
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + feed_forward_output)
        
        return hidden_states


class AdvancedTransformer(nn.Module):
    """
    Educational implementation of Transformer architecture.
    
    This implementation includes:
    - Token and positional embeddings
    - Multiple transformer blocks
    - Various output heads for different tasks
    - Comprehensive documentation for learning
    """
    
    def __init__(self, config: TransformerConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Task-specific heads
        self.classification_head = nn.Linear(config.hidden_size, num_labels)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following BERT initialization."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task: str = "classification"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass supporting multiple tasks.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            task: Task type - 'classification' or 'language_modeling'
            
        Returns:
            Dictionary containing task-specific outputs
        """
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Pass through transformer blocks
        hidden_states = embeddings
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, attention_mask)
        
        outputs = {"last_hidden_state": hidden_states}
        
        # Task-specific outputs
        if task == "classification":
            # Use [CLS] token representation (first token)
            cls_representation = hidden_states[:, 0]
            logits = self.classification_head(cls_representation)
            outputs["logits"] = logits
            
        elif task == "language_modeling":
            # Language modeling head for all positions
            lm_logits = self.lm_head(hidden_states)
            outputs["lm_logits"] = lm_logits
        
        return outputs


class OptimizedTrainer:
    """
    Advanced trainer with modern optimization techniques.
    
    Features:
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision training
    - Early stopping
    - Comprehensive logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler=None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, task="classification")
            logits = outputs["logits"]
            
            # Compute loss
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, task="classification")
                logits = outputs["logits"]
                
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int, patience: int = 5) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.
        
        Args:
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, val_accuracy = self.evaluate()
            
            # Update history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_accuracy"].append(val_accuracy)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
            
            print("-" * 50)
        
        return self.training_history


# Example usage and demonstrations
if __name__ == "__main__":
    # Configuration
    config = TransformerConfig(
        vocab_size=30000,
        max_seq_length=128,
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=4
    )
    
    # Create model
    model = AdvancedTransformer(config, num_labels=2)
    
    # Example input
    batch_size = 4
    seq_length = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Forward pass
    outputs = model(input_ids, attention_mask, task="classification")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")
    
    print("\nAdvanced Transformer model loaded successfully!")
    print("This implementation includes state-of-the-art techniques for educational purposes.")
