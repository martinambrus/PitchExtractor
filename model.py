"""Model architectures for F0 estimation."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with optional dropout."""

    def __init__(self, dim: int, dropout: float = 0.0, max_len: int = 10000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, dim)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ConformerFeedForward(nn.Module):
    """Position-wise feed-forward module used inside Conformer blocks."""

    def __init__(self, dim: int, multiplier: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(dim * multiplier)
        self.layer_norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.layer_norm(x))


class ConformerConvModule(nn.Module):
    """Depthwise separable convolutional module from the Conformer paper."""

    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Conformer convolution kernel size must be odd to preserve sequence length")

        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, dim)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch, dim, time)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    """Standard Conformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_multiplier: float = 4.0,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff_scale = 0.5
        self.ffn1 = ConformerFeedForward(dim, ff_multiplier, dropout)
        self.ffn2 = ConformerFeedForward(dim, ff_multiplier, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(dim, conv_kernel_size, dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Feed-forward module 1
        x = x + self.ff_scale * self.ffn1(x)

        # Multi-head self-attention
        attn_input = self.attn_norm(x)
        attn_output, _ = self.self_attn(
            attn_input, attn_input, attn_input, key_padding_mask=key_padding_mask
        )
        x = x + self.attn_dropout(attn_output)

        # Convolutional module
        x = x + self.conv_module(x)

        # Feed-forward module 2
        x = x + self.ff_scale * self.ffn2(x)

        return self.final_norm(x)


class PitchConformer(nn.Module):
    """Conformer-based backbone for pitch regression and voiced/unvoiced detection."""

    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        ff_multiplier: float = 4.0,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        final_dropout: float = 0.1,
        num_pitch_bins: int = 1,
        **extra_config: Dict,
    ) -> None:
        super().__init__()
        if "num_class" in extra_config:
            num_pitch_bins = int(extra_config.pop("num_class"))
        if num_pitch_bins < 1:
            raise ValueError("num_pitch_bins must be >= 1")

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    dim=d_model,
                    num_heads=num_heads,
                    ff_multiplier=ff_multiplier,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        head_hidden = max(d_model // 2, 64)
        self.pitch_head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(final_dropout),
            nn.Linear(head_hidden, num_pitch_bins),
        )
        self.voicing_head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(final_dropout),
            nn.Linear(head_hidden, 1),
        )

        self.reset_parameters()

    @staticmethod
    def _lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        range_tensor = torch.arange(max_length, device=lengths.device).unsqueeze(0)
        return range_tensor >= lengths.unsqueeze(1)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
        for module in self.pitch_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.voicing_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """Run the Conformer encoder.

        Args:
            x: Tensor shaped ``(batch, time, mel_bins)``.
            lengths: Optional tensor of input lengths. When provided, padding
                positions are ignored by the self-attention layers.

        Returns:
            Tuple of tensors ``(pitch, silence_logits)`` where ``pitch`` has
            shape ``(batch, time, num_pitch_bins)`` and ``silence_logits`` has
            shape ``(batch, time)``.
        """

        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = self._lengths_to_mask(lengths, x.size(1))

        x = self.input_projection(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.output_norm(x)

        pitch = self.pitch_head(x)
        silence_logits = self.voicing_head(x).squeeze(-1)
        return pitch, silence_logits


class JDCNet(nn.Module):
    """
    Joint Detection and Classification Network model for singing voice melody.
    """

    def __init__(self, num_class=722, leaky_relu_slope=0.01):
        super().__init__()
        self.num_class = num_class

        # input = (b, 1, 31, 513), b = batch size
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False),  # out: (b, 64, 31, 513)
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),  # (b, 64, 31, 513)
        )

        # res blocks
        self.res_block1 = ResBlock(in_channels=64, out_channels=128)  # (b, 128, 31, 128)
        self.res_block2 = ResBlock(in_channels=128, out_channels=192)  # (b, 192, 31, 32)
        self.res_block3 = ResBlock(in_channels=192, out_channels=256)  # (b, 256, 31, 8)

        # pool block
        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),  # (b, 256, 31, 2)
            nn.Dropout(p=0.5),
        )

        # maxpool layers (for auxiliary network inputs)
        # in = (b, 128, 31, 513) from conv_block, out = (b, 128, 31, 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 40))
        # in = (b, 128, 31, 128) from res_block1, out = (b, 128, 31, 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 20))
        # in = (b, 128, 31, 32) from res_block2, out = (b, 128, 31, 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 10))

        # in = (b, 640, 31, 2), out = (b, 256, 31, 2)
        self.detector_conv = nn.Sequential(
            nn.Conv2d(640, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Dropout(p=0.5),
        )

        # input: (b, 31, 512) - resized from (b, 256, 31, 2)
        self.bilstm_classifier = nn.LSTM(
            input_size=512,
            hidden_size=256,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )  # (b, 31, 512)

        # input: (b, 31, 512) - resized from (b, 256, 31, 2)
        self.bilstm_detector = nn.LSTM(
            input_size=512,
            hidden_size=256,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )  # (b, 31, 512)

        # input: (b * 31, 512)
        self.classifier = nn.Linear(in_features=512, out_features=self.num_class)  # (b * 31, num_class)

        # input: (b * 31, 512)
        self.detector = nn.Linear(in_features=512, out_features=2)  # (b * 31, 2) - binary classifier

        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        """
        Returns:
            classification_prediction, detection_prediction
            sizes: (b, 31, 722), (b, 31, 2)
        """
        seq_len = x.shape[-2]
        ###############################
        # forward pass for classifier #
        ###############################
        convblock_out = self.conv_block(x)

        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)
        poolblock_out = self.pool_block(resblock3_out)

        # (b, 256, 31, 2) => (b, 31, 256, 2) => (b, 31, 512)
        classifier_out = poolblock_out.permute(0, 2, 1, 3).contiguous().view((-1, seq_len, 512))
        classifier_out, _ = self.bilstm_classifier(classifier_out)  # ignore the hidden states

        classifier_out = classifier_out.contiguous().view((-1, 512))  # (b * 31, 512)
        classifier_out = self.classifier(classifier_out)
        classifier_out = classifier_out.view((-1, seq_len, self.num_class))  # (b, 31, num_class)

        #############################
        # forward pass for detector #
        #############################
        mp1_out = self.maxpool1(convblock_out)
        mp2_out = self.maxpool2(resblock1_out)
        mp3_out = self.maxpool3(resblock2_out)

        # out = (b, 640, 31, 2)
        concat_out = torch.cat((mp1_out, mp2_out, mp3_out, poolblock_out), dim=1)
        detector_out = self.detector_conv(concat_out)

        # (b, 256, 31, 2) => (b, 31, 256, 2) => (b, 31, 512)
        detector_out = detector_out.permute(0, 2, 1, 3).contiguous().view((-1, seq_len, 512))
        detector_out, _ = self.bilstm_detector(detector_out)  # (b, 31, 512)

        detector_out = detector_out.contiguous().view((-1, 512))
        detector_out = self.detector(detector_out)
        detector_out = detector_out.view((-1, seq_len, 2)).sum(axis=-1)  # binary classifier - (b, 31, 2)

        # sizes: (b, 31, 722), (b, 31, 2)
        # classifier output consists of predicted pitch classes per frame
        # detector output consists of: (isvoice, notvoice) estimates per frame
        return classifier_out, detector_out

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
            for p in m.parameters():
                if p.data is None:
                    continue

                if len(p.shape) >= 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.normal_(p.data)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope=0.01):
        super().__init__()
        self.downsample = in_channels != out_channels

        # BN / LReLU / MaxPool layer before the conv layer - see Figure 1b in the paper
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # apply downsampling on the y axis only
        )

        # conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )

        # 1 x 1 convolution layer to match the feature dimensions
        self.conv1by1 = None
        if self.downsample:
            self.conv1by1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pre_conv(x)
        if self.downsample:
            x = self.conv(x) + self.conv1by1(x)
        else:
            x = self.conv(x) + x
        return x


MODEL_REGISTRY: Dict[str, nn.Module] = {
    "conformer": PitchConformer,
    "jdcnet": JDCNet,
}


def build_model(config: Optional[Dict]) -> nn.Module:
    """Instantiate a model from a configuration dictionary."""

    config = config.copy() if config else {}
    name = config.pop("name", "conformer").lower()
    if name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{name}'. Available models: {supported}")
    model_cls = MODEL_REGISTRY[name]
    return model_cls(**config)
