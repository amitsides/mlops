# config_parser.py
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class LayerConfig:
    type: str
    units: int
    activation: str
    rate: float = 0.0

@dataclass
class ModelConfig:
    name: str
    layers: List[LayerConfig]

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    optimizer: str
    loss: str
    metrics: List[str]

@dataclass
class DataConfig:
    train_data: str
    valid_data: str
    test_data: str

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

def load_config(config_file: str) -> Config:
    import json
    from pathlib import Path

    config_path = Path(config_file)
    if config_path.suffix == ".json":
        with open(config_path, "r") as f:
            config_data = json.load(f)
    elif config_path.suffix == ".yaml":
        import yaml
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    else:
        raise ValueError("Invalid configuration file format. Use .json or .yaml")

    model_config = ModelConfig(
        name=config_data["model"]["name"],
        layers=[LayerConfig(**layer) for layer in config_data["model"]["layers"]],
    )

    training_config = TrainingConfig(
        epochs=config_data["training"]["epochs"],
        batch_size=config_data["training"]["batch_size"],
        optimizer=config_data["training"]["optimizer"],
        loss=config_data["training"]["loss"],
        metrics=config_data["training"]["metrics"],
    )

    data_config = DataConfig(
        train_data=config_data["data"]["train_data"],
        valid_data=config_data["data"]["valid_data"],
        test_data=config_data["data"]["test_data"],
    )

    return Config(
        model=model_config,
        training=training_config,
        data=data_config,
    )