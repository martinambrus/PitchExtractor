"""
Implementation of model from:
Kum et al. - "Joint Detection and Classification of Singing Voice Melody Using
Convolutional Recurrent Neural Networks" (2019)
Link: https://www.semanticscholar.org/paper/Joint-Detection-and-Classification-of-Singing-Voice-Kum-Nam/60a2ad4c7db43bace75805054603747fcd062c0d
"""
import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import nn


class JDCNet(nn.Module):
    """
    Joint Detection and Classification Network model for singing voice melody.
    """
    def __init__(self, num_class=722, leaky_relu_slope=0.01, sequence_model_config=None):
        super().__init__()
        self.num_class = num_class
        sequence_model_config = sequence_model_config or {}

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

        sequence_model_config.setdefault('input_size', 512)
        self.sequence_classifier = SequenceModel(**sequence_model_config)
        self.sequence_detector = SequenceModel(**sequence_model_config)

        classifier_dim = self.sequence_classifier.output_dim
        detector_dim = self.sequence_detector.output_dim

        # input: (b * 31, classifier_dim)
        self.classifier = nn.Linear(in_features=classifier_dim, out_features=self.num_class)  # (b * 31, num_class)

        # input: (b * 31, detector_dim)
        self.detector = nn.Linear(in_features=detector_dim, out_features=2)  # (b * 31, 2) - binary classifier

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
        classifier_out = self.sequence_classifier(classifier_out)

        classifier_out = classifier_out.contiguous().view((-1, classifier_out.shape[-1]))  # (b * 31, hidden)
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
        detector_out = self.sequence_detector(detector_out)  # (b, 31, hidden)

        detector_out = detector_out.contiguous().view((-1, detector_out.shape[-1]))
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
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias=False),
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


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding compatible with batch-first inputs."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class SequenceModel(nn.Module):
    """Flexible temporal modeling block supporting BiLSTM and Transformer backends."""

    def __init__(
        self,
        input_size: int,
        model_type: str = "bilstm",
        hidden_size: int = 384,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        max_len: int = 2000,
    ):
        super().__init__()
        self.model_type = model_type.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        if self.model_type == "bilstm":
            lstm_dropout = dropout if num_layers > 1 else 0.0
            self.model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=lstm_dropout,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self._output_dim = hidden_size * (2 if bidirectional else 1)
        elif self.model_type == "transformer":
            self.pos_encoding = SinusoidalPositionalEncoding(input_size, max_len=max_len)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.layer_norm = nn.LayerNorm(input_size)
            self._output_dim = input_size
        else:
            raise ValueError(f"Unsupported sequence model type: {model_type}")

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type == "bilstm":
            x, _ = self.model(x)
            return x
        if self.model_type == "transformer":
            x = self.layer_norm(self.pos_encoding(x))
            return self.model(x)
        raise RuntimeError("Invalid sequence model configuration")


class ConvBNAct(nn.Module):
    """Utility block combining convolution, normalisation, and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] = (3, 3),
        stride: Sequence[int] = (1, 1),
        dilation: Sequence[int] = (1, 1),
        activation_slope: float = 0.01,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(kernel_size) != 2:
            raise ValueError("kernel_size must be a sequence of length 2")
        if len(stride) != 2 or len(dilation) != 2:
            raise ValueError("stride and dilation must have length 2")

        padding = (
            (kernel_size[0] // 2) * dilation[0],
            (kernel_size[1] // 2) * dilation[1],
        )

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(activation_slope, inplace=True),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HarmonicEnergyHead(nn.Module):
    """Small MLP head that predicts per-frame harmonic or energy weights."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RMVPEInspiredNet(nn.Module):
    """Pitch model with a multi-branch convolutional backbone and harmonic/energy heads.

    The design borrows ideas from the RMVPE architecture by combining
    multi-resolution convolutional branches with dedicated heads that estimate
    harmonic strength and frame energy. These auxiliary estimates are used to
    stabilise the regression output on noisy or expressive singing voice data.
    """

    def __init__(
        self,
        num_class: int = 1,
        sequence_model_config: Optional[dict] = None,
        base_channels: int = 64,
        branch_channels: int = 64,
        branch_dilations: Iterable[int] = (1, 2, 4),
        branch_time_kernel_sizes: Iterable[int] = (3, 5, 9),
        freq_kernel_size: int = 5,
        dropout: float = 0.1,
        harmonic_head_hidden: int = 128,
        energy_head_hidden: int = 128,
    ) -> None:
        super().__init__()

        self.num_class = num_class
        sequence_model_config = dict(sequence_model_config or {})

        dilations = tuple(branch_dilations)
        time_kernels = tuple(branch_time_kernel_sizes)
        if len(dilations) != len(time_kernels):
            raise ValueError("branch_dilations and branch_time_kernel_sizes must match in length")

        self.stem = nn.Sequential(
            ConvBNAct(1, base_channels, kernel_size=(freq_kernel_size, 5), dropout=dropout),
            ConvBNAct(base_channels, base_channels, kernel_size=(3, 3), dropout=dropout),
        )

        branches = []
        for dilation, kernel_t in zip(dilations, time_kernels):
            branch = nn.Sequential(
                ConvBNAct(
                    base_channels,
                    branch_channels,
                    kernel_size=(freq_kernel_size, kernel_t),
                    dilation=(1, dilation),
                    dropout=dropout,
                ),
                ConvBNAct(
                    branch_channels,
                    branch_channels,
                    kernel_size=(3, 3),
                    dilation=(1, 1),
                    dropout=dropout,
                ),
            )
            branches.append(branch)
        self.branches = nn.ModuleList(branches)

        fusion_channels = branch_channels * len(self.branches)
        self.fusion = nn.Sequential(
            ConvBNAct(fusion_channels, fusion_channels, kernel_size=(3, 3), dropout=dropout),
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.freq_pool = nn.AdaptiveAvgPool2d((None, 1))
        sequence_model_config["input_size"] = fusion_channels
        self.sequence_model = SequenceModel(**sequence_model_config)

        sequence_dim = self.sequence_model.output_dim
        self.pitch_head = nn.Linear(sequence_dim, num_class)
        self.harmonic_head = HarmonicEnergyHead(sequence_dim, harmonic_head_hidden, dropout=dropout)
        self.energy_head = HarmonicEnergyHead(sequence_dim, energy_head_hidden, dropout=dropout)

        self.apply(JDCNet.init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, seq_len, _ = x.shape
        stem_out = self.stem(x)

        branch_features = [branch(stem_out) for branch in self.branches]
        fused = torch.cat(branch_features, dim=1)
        fused = self.fusion(fused)
        fused = self.freq_pool(fused).squeeze(-1)  # (B, C, T)
        fused = fused.transpose(1, 2).contiguous()  # (B, T, C)

        temporal_features = self.sequence_model(fused)

        harmonic_logits = self.harmonic_head(temporal_features)
        harmonic_weight = torch.sigmoid(harmonic_logits).unsqueeze(-1)

        energy_logits = self.energy_head(temporal_features)
        energy_weight = torch.sigmoid(energy_logits).unsqueeze(-1)

        pitch_raw = self.pitch_head(temporal_features)
        pitch_pred = pitch_raw * harmonic_weight * energy_weight

        silence_pred = energy_logits
        silence_pred = silence_pred.view(batch, seq_len)

        return pitch_pred, silence_pred


def build_pitch_model(model_config: dict) -> nn.Module:
    """Factory that instantiates a pitch model from a configuration dictionary."""

    architecture = model_config.get("architecture", "jdc").lower()
    sequence_model_config = model_config.get("sequence_model", {})
    num_class = model_config.get("num_class", 1)

    if architecture in {"rmvpe", "multi_branch", "harmonic_energy"}:
        rmvpe_params = model_config.get("rmvpe_params", {})
        return RMVPEInspiredNet(
            num_class=num_class,
            sequence_model_config=sequence_model_config,
            **rmvpe_params,
        )

    if architecture == "jdc":
        return JDCNet(num_class=num_class, sequence_model_config=sequence_model_config)

    raise ValueError(f"Unsupported model architecture: {architecture}")
