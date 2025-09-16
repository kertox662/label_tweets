from dataclasses import dataclass
from typing import List

@dataclass
class SupervisedTrainingConfig:
    # Model params
    model_name: str | None = None
    checkpoint_name: str | None = None
    soft_labels: bool = True
    class_weight: List[int] | None = None
    freeze_encoder: bool = True
    hidden_dim: int = 128
    dropout_p: float = 0.3

    # Training params
    learning_rate: float = 1e-4
    max_epochs: int = 1
    accumulate_grad_batches: int = 8
    stopping_patience: int = 2
    warmup_ratio: float = 0.1

    # Metrics
    logs_dir: str = "tb_logs"
    logs_name: str = "classifier"

    # Data params
    train_data: str = "data/tweets/train_master.csv"
    test_data: str = "data/tweets/test_master.csv"
    batch_size: int = 32
