from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SupervisedTrainingConfig:
    # Model params
    model_name: str | None = None
    checkpoint_name: str | None = None # Where to start training from, if not None
    class_weight: List[int] | None = None
    dropout_p: float = 0.3
    trials: int = 20
    cross_val_folds: Optional[int] = None

    # Training params
    learning_rate: float = 1e-4
    max_epochs: int = 1
    accumulate_grad_batches: int = 8
    stopping_patience: int = 2
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
