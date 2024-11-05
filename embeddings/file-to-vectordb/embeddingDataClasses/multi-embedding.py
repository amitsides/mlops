from dataclasses import dataclass, field
import torch
from typing import List, Optional, Union
import numpy as np

@dataclass
class BaseEmbeddingConfig:
    model_name: str
    max_length: int
    batch_size: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TextEmbeddingConfig(BaseEmbeddingConfig):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 512
    batch_size: int = 32
    pooling_strategy: str = "mean"

@dataclass
class GraphEmbeddingConfig(BaseEmbeddingConfig):
    model_name: str = "dgl/graphsage-pubmed"
    max_nodes: int = 1000
    node_features: int = 128
    graph_conv_layers: List[int] = field(default_factory=lambda: [64, 64])
    readout_method: str = "mean"

@dataclass
class MoleculeEmbeddingConfig(BaseEmbeddingConfig):
    model_name: str = "deepchem/mol2vec"
    featurizer: str = "ECFP"
    radius: int = 2
    size: int = 1024
    chirality: bool = True

@dataclass
class TimeSeriesEmbeddingConfig(BaseEmbeddingConfig):
    model_name: str = "keras/tcn"
    sequence_length: int = 100
    n_features: int = 1
    kernel_size: int = 3
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

@dataclass
class Model3DEmbeddingConfig(BaseEmbeddingConfig):
    model_name: str = "pointnet2"
    num_points: int = 1024
    normal_channel: bool = True
    feature_transform: bool = True

@dataclass
class MultiModalEmbeddingConfig:
    text_config: TextEmbeddingConfig
    graph_config: Optional[GraphEmbeddingConfig] = None
    molecule_config: Optional[MoleculeEmbeddingConfig] = None
    time_series_config: Optional[TimeSeriesEmbeddingConfig] = None
    model_3d_config: Optional[Model3DEmbeddingConfig] = None
    fusion_method: str = "concatenate"
    output_dim: int = 512

@dataclass
class EmbeddingResult:
    embedding: Union[np.ndarray, torch.Tensor]
    metadata: dict = field(default_factory=dict)

@dataclass
class EmbeddingPipeline:
    config: MultiModalEmbeddingConfig
    
    def embed(self, data: Union[str, np.ndarray, torch.Tensor, object]) -> EmbeddingResult:
        # Implementation for embedding generation
        pass