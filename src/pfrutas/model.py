from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union

import torch
from torch import nn

from .features import FeatureStandardizer


@dataclass
class ModelConfig:
    input_dim: int
    num_classes: int
    projection_dim: int = 160
    hidden_dim: int = 96
    num_layers: int = 2
    dropout: float = 0.35
    bidirectional: bool = True


LANDMARK_MODEL_FAMILY = "landmark_bilstm"
ASL_TRANSFER_MODEL_FAMILY = "asl_citizen_stgcn_fusion"


@dataclass
class ASLTransferModelConfig(ModelConfig):
    """Configuración versionada de la fusión LSEC + encoder ASL Citizen."""

    model_family: str = ASL_TRANSFER_MODEL_FAMILY
    transfer_backbone: str = "asl_citizen_stgcn_v1"
    graph_sequence_length: int = 128
    fusion_dim: int = 96
    unfreeze_backbone_blocks: int = 1
    unfreeze_backbone_epoch: int = 6
    backbone_learning_rate: float = 1e-5

    def __post_init__(self) -> None:
        # El checkpoint oficial fue entrenado con T=128; no es un hiperparámetro
        # libre. Fallar al leer la configuración evita crear un modelo que luego
        # recibiría inevitablemente un grafo de forma incompatible.
        from .asl_transfer import ASL_GRAPH_SEQUENCE_LENGTH

        if self.graph_sequence_length != ASL_GRAPH_SEQUENCE_LENGTH:
            raise ValueError(
                "graph_sequence_length debe ser exactamente "
                f"{ASL_GRAPH_SEQUENCE_LENGTH} para ASL Citizen v1."
            )


# ``typing.TypeAlias`` y la sintaxis ``X | Y`` son de Python 3.10. El runtime
# experimental JetPack 5 usa Python 3.8, por lo que se conserva el mismo
# contrato con ``Union`` sin afectar los entornos modernos de entrenamiento.
ModelConfiguration = Union[ModelConfig, ASLTransferModelConfig]


# Los defaults de ModelConfig se mantienen para poder reconstruir checkpoints
# históricos. Los entrenamientos nuevos pueden escoger una arquitectura de forma
# explícita mediante model_config_for_preset.
LEGACY_MODEL_PRESET = "legacy_v1"
COMPACT_MODEL_PRESET = "compact_v2"
ASL_TRANSFER_MODEL_PRESET = "asl_citizen_stgcn_fusion_v1"
DEFAULT_MODEL_PRESET = COMPACT_MODEL_PRESET

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    LEGACY_MODEL_PRESET: {
        "projection_dim": 160,
        "hidden_dim": 96,
        "num_layers": 2,
        "dropout": 0.35,
        "bidirectional": True,
    },
    COMPACT_MODEL_PRESET: {
        "projection_dim": 80,
        "hidden_dim": 40,
        "num_layers": 1,
        "dropout": 0.25,
        "bidirectional": True,
    },
    ASL_TRANSFER_MODEL_PRESET: {
        "model_family": ASL_TRANSFER_MODEL_FAMILY,
        "projection_dim": 80,
        "hidden_dim": 40,
        "num_layers": 1,
        "dropout": 0.25,
        "bidirectional": True,
        "graph_sequence_length": 128,
        "fusion_dim": 96,
        "unfreeze_backbone_blocks": 1,
        "unfreeze_backbone_epoch": 6,
        "backbone_learning_rate": 1e-5,
    },
}


def model_config_for_preset(
    input_dim: int,
    num_classes: int,
    preset: str = DEFAULT_MODEL_PRESET,
) -> ModelConfiguration:
    """Construye una configuración reproducible sin alterar el formato legado."""
    try:
        parameters = MODEL_PRESETS[preset]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(
            f"Preset de modelo desconocido: {preset!r}. Opciones: {available}."
        ) from exc
    parameters = dict(parameters)
    family = parameters.pop("model_family", LANDMARK_MODEL_FAMILY)
    if family == LANDMARK_MODEL_FAMILY:
        return ModelConfig(
            input_dim=input_dim,
            num_classes=num_classes,
            **parameters,
        )
    if family == ASL_TRANSFER_MODEL_FAMILY:
        return ASLTransferModelConfig(
            input_dim=input_dim,
            num_classes=num_classes,
            model_family=family,
            **parameters,
        )
    raise ValueError(f"Familia de modelo desconocida en {preset!r}: {family!r}.")


class SignLSTM(nn.Module):
    """BiLSTM ligera con atención temporal para clasificación de señas."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(
            nn.Linear(config.input_dim, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size=config.projection_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        recurrent_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.attention = nn.Sequential(
            nn.Linear(recurrent_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(recurrent_dim * 2),
            nn.Dropout(config.dropout),
            nn.Linear(recurrent_dim * 2, config.num_classes),
        )

    def encode(
        self,
        inputs: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        projected = self.projection(inputs)
        temporal, _ = self.lstm(projected)
        attention = torch.softmax(self.attention(temporal).squeeze(-1), dim=1)
        attended = torch.sum(temporal * attention.unsqueeze(-1), dim=1)
        maximum = torch.amax(temporal, dim=1)
        encoding = torch.cat((attended, maximum), dim=-1)
        if return_attention:
            return encoding, attention
        return encoding

    def forward(
        self,
        inputs: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(inputs, return_attention=return_attention)
        if return_attention:
            features, attention = encoded
            return self.classifier(features), attention
        return self.classifier(encoded)


class ASLTransferFusion(SignLSTM):
    """Fusiona landmarks LSEC con un encoder ST-GCN preentrenado en ASL.

    La cabeza ASL original no se instancia. El encoder externo aprende patrones
    de configuración/movimiento; las etiquetas de ASL nunca se mapean a frutas.
    """

    def __init__(self, config: ASLTransferModelConfig) -> None:
        from .asl_transfer import ASLCitizenSTGCN

        super().__init__(config)
        self.config = config
        recurrent_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        landmark_dim = recurrent_dim * 2
        self.asl_encoder = ASLCitizenSTGCN()
        self.landmark_projection = nn.Sequential(
            nn.LayerNorm(landmark_dim),
            nn.Linear(landmark_dim, config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.5),
        )
        self.graph_projection = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.5),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.fusion_dim * 2, config.fusion_dim // 2),
            nn.GELU(),
            nn.Linear(config.fusion_dim // 2, 1),
        )
        # Al inicio se confía 75 % en la rama local ya validada. El gate puede
        # aumentar el aporte ASL si los folds internos encuentran señal útil.
        nn.init.zeros_(self.fusion_gate[-1].weight)
        nn.init.constant_(self.fusion_gate[-1].bias, 1.0986122886681098)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.fusion_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim, config.num_classes),
        )
        self._trainable_backbone_blocks = 0
        self.set_trainable_backbone_blocks(0)
        self.pretraining_metadata: dict[str, Any] = {}

    def set_trainable_backbone_blocks(self, blocks: int) -> None:
        networks = self.asl_encoder.st_gcn_networks
        if not 0 <= blocks <= len(networks):
            raise ValueError(
                f"blocks debe estar entre 0 y {len(networks)}; se recibió {blocks}."
            )
        for parameter in self.asl_encoder.parameters():
            parameter.requires_grad_(False)
        if blocks:
            for block in networks[-blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.asl_encoder.edge_importance[-blocks:]:
                parameter.requires_grad_(True)
        self._trainable_backbone_blocks = blocks

    def configure_epoch(self, epoch: int) -> None:
        blocks = (
            self.config.unfreeze_backbone_blocks
            if epoch >= self.config.unfreeze_backbone_epoch
            else 0
        )
        if blocks != self._trainable_backbone_blocks:
            self.set_trainable_backbone_blocks(blocks)

    def optimizer_parameter_groups(self, learning_rate: float) -> list[dict[str, Any]]:
        head = [
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("asl_encoder.")
        ]
        backbone = list(self.asl_encoder.parameters())
        return [
            {"params": head, "lr": learning_rate},
            {"params": backbone, "lr": self.config.backbone_learning_rate},
        ]

    def train(self, mode: bool = True) -> "ASLTransferFusion":
        super().train(mode)
        # Las estadísticas BatchNorm preentrenadas nunca se recalculan con el
        # dataset diminuto. Los bloques abiertos conservan dropout/gradientes,
        # pero sus BatchNorm permanecen en evaluación.
        self.asl_encoder.eval()
        if mode and self._trainable_backbone_blocks:
            for block in self.asl_encoder.st_gcn_networks[
                -self._trainable_backbone_blocks :
            ]:
                block.train(True)
                for module in block.modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        module.eval()
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        graph: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if graph.ndim != 4 or graph.shape[1:] != (
            2,
            self.config.graph_sequence_length,
            27,
        ):
            raise ValueError(
                "El grafo ASL debe tener forma [batch, 2, "
                f"{self.config.graph_sequence_length}, 27]."
            )
        landmark_features, attention = super().encode(
            inputs,
            return_attention=True,
        )
        landmark_embedding = self.landmark_projection(landmark_features)
        graph_embedding = self.graph_projection(self.asl_encoder(graph))
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat((landmark_embedding, graph_embedding), dim=-1))
        )
        fused = gate * landmark_embedding + (1.0 - gate) * graph_embedding
        logits = self.classifier(fused)
        if return_attention:
            return logits, attention
        return logits


SignModel = Union[SignLSTM, ASLTransferFusion]


def is_asl_transfer_config(config: ModelConfiguration) -> bool:
    return isinstance(config, ASLTransferModelConfig)


def build_model(
    config: ModelConfiguration,
    pretrained_checkpoint: Path | None = None,
) -> SignModel:
    if isinstance(config, ASLTransferModelConfig):
        from .asl_transfer import (
            ASL_CITIZEN_CHECKPOINT_SHA256,
            load_asl_citizen_encoder,
        )

        model = ASLTransferFusion(config)
        if pretrained_checkpoint is not None:
            model.pretraining_metadata = load_asl_citizen_encoder(
                model.asl_encoder,
                pretrained_checkpoint,
                expected_sha256=ASL_CITIZEN_CHECKPOINT_SHA256,
            )
        return model
    if pretrained_checkpoint is not None:
        raise ValueError(
            "--pretrained-checkpoint solo se admite con el preset de transferencia ASL."
        )
    return SignLSTM(config)


def save_checkpoint(
    path: Path,
    model: SignModel,
    optimizer: torch.optim.Optimizer | None,
    scaler: FeatureStandardizer,
    labels: list[str],
    sequence_length: int,
    fold: int,
    epoch: int,
    metrics: dict[str, Any],
    preprocessing: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "model_config": asdict(model.config),
        "feature_scaler": scaler.to_dict(),
        "labels": labels,
        "sequence_length": sequence_length,
        "fold": fold,
        "epoch": epoch,
        "metrics": metrics,
        "preprocessing": preprocessing,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    device: str,
    *,
    weights_only: bool = False,
    retain_training_state: bool = True,
) -> tuple[SignModel, FeatureStandardizer, dict[str, Any]]:
    # Cargar primero en CPU evita subir los estados de AdamW a la GPU durante
    # evaluación/inferencia; solamente los pesos del modelo se mueven al device.
    # Los checkpoints históricos requieren ``weights_only=False``. Los bundles
    # de despliegue saneados pueden (y deben) solicitar la carga restringida.
    payload = torch.load(path, map_location="cpu", weights_only=weights_only)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint inválido {path}: se esperaba un diccionario.")
    try:
        model_state = (
            payload["model_state"]
            if retain_training_state
            else payload.pop("model_state")
        )
    except KeyError as exc:
        raise RuntimeError(f"Checkpoint inválido {path}: falta model_state.") from exc
    if not retain_training_state:
        # El optimizador no participa en evaluación y puede duplicar
        # temporalmente el uso de RAM en dispositivos de memoria compartida.
        payload.pop("optimizer_state", None)
    raw_config = payload["model_config"]
    if raw_config.get("model_family") == ASL_TRANSFER_MODEL_FAMILY:
        config: ModelConfiguration = ASLTransferModelConfig(**raw_config)
    else:
        config = ModelConfig(**raw_config)
    model = build_model(config)
    model.load_state_dict(model_state)
    del model_state
    model.to(device)
    model.eval()
    scaler = FeatureStandardizer.from_dict(payload["feature_scaler"])
    return model, scaler, payload
