"""Neural network architectures for melody extraction.

This module historically implemented the JDCNet architecture from Kum et al.
(2019). While functional, that CNN/LSTM hybrid is now considered outdated
compared to recent Conformer and Transformer style backbones. To provide a
stronger default model we now ship a Conformer-based network that integrates
modern self-attention, depthwise separable convolutions, and lightweight
feed-forward blocks. The original JDCNet implementation is preserved for
backwards compatibility, but new training runs should prefer the Conformer
variant exposed via :func:`build_model`.
"""

import math
from typing import Iterable, Optional, Tuple

import torch
from torch import nn


class JDCNet(nn.Module):
    """Joint Detection and Classification Network model for singing voice melody."""

    def __init__(self, num_class: int = 722, leaky_relu_slope: float = 0.01):
        super().__init__()
        self.num_class = num_class

        # input = (b, 1, 31, 513), b = batch size
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),  # out: (b, 64, 31, 513)
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
        self.classifier = nn.Linear(
            in_features=512, out_features=self.num_class
        )  # (b * 31, num_class)

        # input: (b * 31, 512)
        self.detector = nn.Linear(
            in_features=512, out_features=2
        )  # (b * 31, 2) - binary classifier

        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning pitch classification and voicing logits."""

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
        classifier_out = (
            poolblock_out.permute(0, 2, 1, 3).contiguous().view((-1, seq_len, 512))
        )
        classifier_out, _ = self.bilstm_classifier(
            classifier_out
        )  # ignore the hidden states

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
        detector_out = (
            detector_out.permute(0, 2, 1, 3).contiguous().view((-1, seq_len, 512))
        )
        detector_out, _ = self.bilstm_detector(detector_out)  # (b, 31, 512)

        detector_out = detector_out.contiguous().view((-1, 512))
        detector_out = self.detector(detector_out)
        detector_out = detector_out.view((-1, seq_len, 2)).sum(
            axis=-1
        )  # binary classifier - (b, 31, 2)

        # sizes: (b, 31, 722), (b, 31, 2)
        # classifier output consists of predicted pitch classes per frame
        # detector output consists of: (isvoice, notvoice) estimates per frame
        return classifier_out, detector_out

    @staticmethod
    def init_weights(m: nn.Module) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(x)
        if self.downsample:
            x = self.conv(x) + self.conv1by1(x)
        else:
            x = self.conv(x) + x
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dropout regularisation."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        x = x + self.pe[:, :length]
        return self.dropout(x)


class FeedForwardModule(nn.Module):
    """Position-wise feed-forward module used inside the Conformer block."""

    def __init__(self, dim: int, expansion: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, expansion)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class MultiHeadSelfAttentionModule(nn.Module):
    """Self-attention sub-layer with pre-norm and dropout."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x_norm = self.layer_norm(x)
        attn_output, _ = self.attention(
            x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask
        )
        return self.dropout(attn_output)


class ConformerConvModule(nn.Module):
    """Depthwise convolutional module from the Conformer architecture."""

    def __init__(self, dim: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Conformer convolution kernel size must be odd")

        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return self.dropout(x)


class ConformerBlock(nn.Module):
    """A single Conformer encoder block."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ff_multiplier: float,
        conv_kernel: int,
        dropout: float,
    ):
        super().__init__()
        expansion = int(dim * ff_multiplier)
        self.ff_module1 = FeedForwardModule(dim, expansion, dropout)
        self.self_attention = MultiHeadSelfAttentionModule(dim, heads, dropout)
        self.conv_module = ConformerConvModule(dim, conv_kernel, dropout)
        self.ff_module2 = FeedForwardModule(dim, expansion, dropout)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + 0.5 * self.ff_module1(x)
        x = x + self.self_attention(x, key_padding_mask=key_padding_mask)
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff_module2(x)
        return self.final_layer_norm(x)


class ConformerEncoder(nn.Module):
    """Stack of Conformer blocks."""

    def __init__(
        self,
        num_layers: int,
        dim: int,
        heads: int,
        ff_multiplier: float,
        conv_kernel: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConformerBlock(dim, heads, ff_multiplier, conv_kernel, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


class ConformerPitchNet(nn.Module):
    """Modern Conformer-based backbone for melody extraction."""

    def __init__(
        self,
        num_class: int = 1,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        ff_multiplier: float = 4.0,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        frontend_channels: Iterable[int] = (64, 128, 192),
    ):
        super().__init__()

        channels = list(frontend_channels)
        if not channels:
            raise ValueError("frontend_channels must contain at least one entry")

        conv_layers = []
        in_channels = 1
        for idx, out_channels in enumerate(channels):
            stride = (1, 2) if idx < len(channels) - 1 else (1, 1)
            conv_layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(),
                ]
            )
            in_channels = out_channels

        self.frontend = nn.Sequential(*conv_layers)
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
        )

        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.encoder = ConformerEncoder(
            num_layers=num_layers,
            dim=d_model,
            heads=num_heads,
            ff_multiplier=ff_multiplier,
            conv_kernel=conv_kernel,
            dropout=dropout,
        )
        self.pre_head_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pitch_head = nn.Linear(d_model, num_class)
        self.voicing_head = nn.Linear(d_model, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Args:
            x: Input mel-spectrogram batch shaped ``(B, 1, T, F)``.
            mask: Optional boolean mask marking padded frames (``True`` means the
                frame should be ignored by attention).

        Returns:
            Tuple containing pitch regression/classification predictions with
            shape ``(B, T, num_class)`` and per-frame silence logits with shape
            ``(B, T)``.
        """

        if mask is None:
            mask = x.squeeze(1).sum(dim=-1).eq(0)

        features = self.frontend(x)
        features = self.projection(features)
        features = features.mean(dim=-1).transpose(1, 2)  # (B, T, d_model)

        features = self.positional_encoding(features)
        encoded = self.encoder(features, key_padding_mask=mask)
        encoded = self.dropout(self.pre_head_norm(encoded))

        pitch = self.pitch_head(encoded)
        silence_logits = self.voicing_head(encoded).squeeze(-1)

        return pitch, silence_logits


def build_model(model_config: Optional[dict] = None) -> nn.Module:
    """Factory function that builds the requested melody extraction model.

    Parameters in ``model_config``:

    ``architecture`` (str):
        Either ``"conformer"`` (default) or ``"jdcnet"`` to select the legacy
        architecture.
    ``num_class`` (int):
        Number of output classes for the pitch head. ``1`` enables regression.
    Remaining keys are forwarded to the model constructor.
    """

    config = dict(model_config or {})
    architecture = config.pop("architecture", "conformer").lower()
    num_class = config.pop("num_class", 1)

    if architecture == "jdcnet":
        return JDCNet(num_class=num_class, **config)

    return ConformerPitchNet(num_class=num_class, **config)


__all__ = ["JDCNet", "ResBlock", "ConformerPitchNet", "build_model"]
