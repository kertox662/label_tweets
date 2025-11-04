from dataclasses import dataclass
from typing import List, Optional

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
    trials: int = 20
    cross_val_folds: Optional[int] = None

    # Training params
    learning_rate: float = 1e-4
    max_epochs: int = 1
    accumulate_grad_batches: int = 8
    stopping_patience: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Metrics
    logs_dir: str = "tb_logs"
    logs_name: str = "classifier"

    # Data params
    train_data: str = "data/tweets/train_master.csv"
    test_data: str = "data/tweets/test_master.csv"
    train_with_test_data: str = "data/tweets/train_with_test.csv"
    all_data: str = "data/tweets/all_tweets.csv"
    batch_size: int = 32

    global_seed: int = 2025
    test_data_seed: int = 2025
    val_data_seed: int = 2025
    val_size: float = 0.2
    test_size: float = 0.2
