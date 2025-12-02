"""
LIMU-BERT: Self-Supervised Learning for IMU Data

Reference: https://github.com/dapowan/LIMU-BERT-Public
Paper: LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications

This implementation is compatible with the original pretrained weights.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit activation function."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    """Layer Normalization with optional bias."""

    def __init__(self, hidden_size: int, variance_epsilon: float = 1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """
    Input Embeddings for LIMU-BERT.

    Projects input features to hidden dimension and adds positional embeddings.
    """

    def __init__(
        self,
        feature_num: int,
        hidden: int,
        seq_len: int,
        emb_norm: bool = True,
    ):
        super().__init__()
        self.lin = nn.Linear(feature_num, hidden)
        self.pos_embed = nn.Embedding(seq_len, hidden)
        self.norm = LayerNorm(hidden) if emb_norm else nn.Identity()
        self.emb_norm = emb_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_num)

        Returns:
            Embedded tensor of shape (batch_size, seq_len, hidden)
        """
        batch_size, seq_len, _ = x.shape
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)

        e = self.lin(x)
        e = e + self.pos_embed(pos)
        return self.norm(e)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Self-Attention mechanism."""

    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.proj_q = nn.Linear(hidden, hidden)
        self.proj_k = nn.Linear(hidden, hidden)
        self.proj_v = nn.Linear(hidden, hidden)
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden)
            mask: Optional attention mask

        Returns:
            Attention output of shape (batch_size, seq_len, hidden)
        """
        batch_size, seq_len, hidden = x.shape

        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden)
        return context


class PositionWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, hidden: int, hidden_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden_ff)
        self.fc2 = nn.Linear(hidden_ff, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Single Transformer Block."""

    def __init__(self, hidden: int, hidden_ff: int, n_heads: int):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(hidden, n_heads)
        self.proj = nn.Linear(hidden, hidden)
        self.norm1 = LayerNorm(hidden)
        self.pwff = PositionWiseFeedForward(hidden, hidden_ff)
        self.norm2 = LayerNorm(hidden)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.attn(x, mask)
        h = self.norm1(x + self.proj(h))
        h = self.norm2(h + self.pwff(h))
        return h


class Transformer(nn.Module):
    """
    Transformer Encoder with weight sharing across layers.

    Note: LIMU-BERT uses weight sharing - same transformer block is applied n_layers times.
    The original implementation puts embed inside transformer for weight saving purposes.
    """

    def __init__(
        self,
        feature_num: int,
        hidden: int,
        hidden_ff: int,
        n_layers: int,
        n_heads: int,
        seq_len: int,
        emb_norm: bool = True,
    ):
        super().__init__()
        self.n_layers = n_layers
        # Embeddings inside transformer (matching original implementation)
        self.embed = Embeddings(feature_num, hidden, seq_len, emb_norm)
        # Single attention layer with weight sharing
        self.attn = MultiHeadedSelfAttention(hidden, n_heads)
        self.proj = nn.Linear(hidden, hidden)
        self.norm1 = LayerNorm(hidden)
        self.pwff = PositionWiseFeedForward(hidden, hidden_ff)
        self.norm2 = LayerNorm(hidden)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_num)
        """
        h = self.embed(x)
        for _ in range(self.n_layers):
            h = self.attn(h, mask)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class LIMUBertModel4Pretrain(nn.Module):
    """
    LIMU-BERT model for pretraining (masked prediction).

    This model is designed for self-supervised pretraining using masked
    reconstruction task similar to BERT's MLM.

    The architecture matches the original implementation for weight loading.
    """

    def __init__(
        self,
        feature_num: int = 6,
        hidden: int = 72,
        hidden_ff: int = 144,
        n_layers: int = 4,
        n_heads: int = 4,
        seq_len: int = 120,
        emb_norm: bool = True,
    ):
        super().__init__()
        self.feature_num = feature_num
        self.hidden = hidden
        self.seq_len = seq_len

        # Transformer includes embeddings (matching original)
        self.transformer = Transformer(
            feature_num, hidden, hidden_ff, n_layers, n_heads, seq_len, emb_norm
        )
        self.fc = nn.Linear(hidden, hidden)
        self.linear = nn.Linear(hidden, hidden)
        self.activ = gelu
        self.norm = LayerNorm(hidden)
        self.decoder = nn.Linear(hidden, feature_num)

    def forward(
        self,
        x: torch.Tensor,
        masked_pos: Optional[torch.Tensor] = None,
        output_embed: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_num)
            masked_pos: Positions to predict of shape (batch_size, n_masked)
            output_embed: If True, return embeddings instead of predictions

        Returns:
            If output_embed: Embeddings of shape (batch_size, seq_len, hidden)
            If masked_pos provided: Predictions of shape (batch_size, n_masked, feature_num)
            Otherwise: Predictions of shape (batch_size, seq_len, feature_num)
        """
        h = self.transformer(x)

        if output_embed:
            return h

        if masked_pos is not None:
            # Extract hidden states at masked positions
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
            h_masked = torch.gather(h, 1, masked_pos)
        else:
            h_masked = h

        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits = self.decoder(h_masked)

        return logits


class LIMUBertEncoder(nn.Module):
    """
    LIMU-BERT Encoder for downstream tasks.

    Extracts features from pretrained LIMU-BERT for classification.
    Uses the same architecture as LIMUBertModel4Pretrain for weight compatibility.
    """

    def __init__(
        self,
        feature_num: int = 6,
        hidden: int = 72,
        hidden_ff: int = 144,
        n_layers: int = 4,
        n_heads: int = 4,
        seq_len: int = 120,
        emb_norm: bool = True,
    ):
        super().__init__()
        self.feature_num = feature_num
        self.hidden = hidden
        self.seq_len = seq_len
        self.output_dim = hidden  # For compatibility with other models

        # Same structure as LIMUBertModel4Pretrain
        self.transformer = Transformer(
            feature_num, hidden, hidden_ff, n_layers, n_heads, seq_len, emb_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, time_steps)
               Note: Will be transposed to (batch_size, time_steps, channels)

        Returns:
            Features of shape (batch_size, hidden)
        """
        # Transpose from (batch, channels, time) to (batch, time, channels)
        if x.dim() == 3 and x.size(1) < x.size(2):
            x = x.transpose(1, 2)

        h = self.transformer(x)

        # Global average pooling over sequence
        h = h.mean(dim=1)  # (batch_size, hidden)

        return h

    def load_pretrained(self, pretrain_model: LIMUBertModel4Pretrain):
        """Load weights from pretrained model."""
        self.transformer.load_state_dict(pretrain_model.transformer.state_dict())


class LIMUBertClassifier(nn.Module):
    """
    LIMU-BERT based classifier for HAR.

    Uses pretrained LIMU-BERT encoder with a classification head.
    """

    def __init__(
        self,
        num_classes: int,
        feature_num: int = 6,
        hidden: int = 72,
        hidden_ff: int = 144,
        n_layers: int = 4,
        n_heads: int = 4,
        seq_len: int = 120,
        emb_norm: bool = True,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")

        self.encoder = LIMUBertEncoder(
            feature_num=feature_num,
            hidden=hidden,
            hidden_ff=hidden_ff,
            n_layers=n_layers,
            n_heads=n_heads,
            seq_len=seq_len,
            emb_norm=emb_norm,
        )

        # Classification head (similar to LIMU-GRU classifier)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes),
        )

        if pretrained_path:
            self._load_pretrained(pretrained_path)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _load_pretrained(self, path: str):
        """Load pretrained weights from LIMU-BERT checkpoint.

        If the pretrained model has different input channels (e.g., 6 channels for ACC+GYRO),
        only loads Transformer layers (attention, projection, feedforward, norm) and skips
        the Embedding layer. This allows using pretrained weights with ACC-only (3 channels) data.
        """
        print(f"Loading LIMU-BERT pretrained weights from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Check if input dimensions match by looking at embedding layer
        pretrained_feature_num = None
        for key, value in state_dict.items():
            if "embed.lin.weight" in key:
                pretrained_feature_num = value.shape[1]  # (hidden, feature_num)
                break

        skip_embedding = False
        if pretrained_feature_num is not None and pretrained_feature_num != self.encoder.feature_num:
            print(f"  Input channel mismatch: pretrained={pretrained_feature_num}, model={self.encoder.feature_num}")
            print(f"  Skipping Embedding layer, loading Transformer layers only")
            skip_embedding = True

        # Extract encoder weights
        # Keys in checkpoint: transformer.embed.*, transformer.attn.*, transformer.proj.*, etc.
        encoder_state = {}
        skipped_keys = []
        for key, value in state_dict.items():
            if key.startswith("transformer."):
                # Skip embedding layer if input dimensions don't match
                if skip_embedding and ".embed." in key:
                    skipped_keys.append(key)
                    continue
                encoder_state[key] = value

        if encoder_state:
            missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
            loaded_count = len(encoder_state)
            print(f"  Loaded {loaded_count} parameters")
            if skipped_keys:
                print(f"  Skipped {len(skipped_keys)} embedding parameters (channel mismatch)")
            if missing:
                # Filter out expected missing keys (embedding when skipped)
                truly_missing = [k for k in missing if not (skip_embedding and ".embed." in k)]
                if truly_missing:
                    print(f"  Missing keys: {truly_missing[:5]}...")
            if unexpected:
                print(f"  Unexpected keys: {unexpected[:5]}...")
        else:
            print("  Warning: No encoder weights found in checkpoint")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch_size, channels, time_steps)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


def create_limu_bert_classifier(
    num_classes: int,
    in_channels: int = 6,
    seq_len: int = 120,
    pretrained_path: Optional[str] = None,
    freeze_encoder: bool = False,
    device: Optional[torch.device] = None,
) -> LIMUBertClassifier:
    """
    Factory function to create LIMU-BERT classifier.

    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels (6 for acc+gyro, 9 for +mag)
        seq_len: Sequence length
        pretrained_path: Path to pretrained weights
        freeze_encoder: Whether to freeze encoder weights
        device: Target device

    Returns:
        LIMUBertClassifier instance
    """
    return LIMUBertClassifier(
        num_classes=num_classes,
        feature_num=in_channels,
        hidden=72,
        hidden_ff=144,
        n_layers=4,
        n_heads=4,
        seq_len=seq_len,
        emb_norm=True,
        pretrained_path=pretrained_path,
        freeze_encoder=freeze_encoder,
        device=device,
    )
