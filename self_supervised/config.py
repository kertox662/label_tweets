from dataclasses import dataclass

@dataclass
class SelfSupervisedTrainingConfig:
    # Model params
    model_name: str | None = None
    checkpoint_name: str | None = None
    learning_rate: float = 1e-4
    loss_scale: float = 7

    # Training params
    max_epochs: int = 1
    max_steps: int = 100_000
    accumulate_grad_batches: int = 8
    stopping_patience: int = 2
    checkpoint_epochs: int = 2

    # Metrics
    validation_interval: int | float = float(0.25)
    logging_interval_steps: int = 200
    logs_dir: str = "tb_logs"
    logs_name: str = "st_ssl"

    # Data params
    data_path: str = "data/tweets/all_tweets.csv"
    batch_size: int = 32

    # Saving params
    save_path: str = "pre_trained_sentence_st"
