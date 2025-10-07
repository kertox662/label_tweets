import os
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from classification.config import SupervisedTrainingConfig
from classification.model import BertweetClassifier
from data.supervised import TweetsDataModule

class SupervisedTrainer:
    def __init__(self, config: SupervisedTrainingConfig):
        self.config = config

        self.model = BertweetClassifier(
            model_name=config.model_name,
            model_checkpoint=config.checkpoint_name,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            freeze_encoder=config.freeze_encoder,
            use_soft_labels=config.soft_labels,
            class_weights=torch.ones(3),
            hidden_dim=config.hidden_dim,
            dropout_p=config.dropout_p,
        )
        
        self.early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.stopping_patience,
        )
        # self.early_stopping = EarlyStopping(
        #     monitor="val_f1_ar",
        #     mode="max",
        #     patience=config.stopping_patience,
        # )

        self.early_stopping2 = EarlyStopping(
            monitor="val_f1_mb",
            mode="max",
            patience=config.stopping_patience,
        )

        self.trainer_args = dict(
            max_epochs=config.max_epochs,
            callbacks=[self.early_stopping, self.early_stopping2],
            logger=TensorBoardLogger(config.logs_dir, name=config.logs_name),
            accumulate_grad_batches=config.accumulate_grad_batches,
        )
    
    def train(self, use_full_test_data=False):
        self.setup_compute_devices()
        trainer = pl.Trainer(**self.trainer_args)
        train_data = pd.read_csv(self.config.train_data)
        train_datamodule = TweetsDataModule(data=train_data, batch_size=self.config.batch_size, random_state=2026, target_col="MB")
        print("Data Read ========================================")
        
        print("Trainer Fitting ========================================")
        trainer.fit(self.model, train_datamodule)
        
        print("Trainer Testing ========================================")
        print("Trainer Test Set ================")
        trainer.test(self.model, train_datamodule)

        
        train_datamodule.test = train_datamodule.train
        print("Trainer Train Set ================")
        trainer.test(self.model, train_datamodule)

        if use_full_test_data:
            print("Full Test Set ================")
            test_data = pd.read_csv(self.config.test_data)
            test_datamodule = TweetsDataModule(data=test_data, batch_size=self.config.batch_size)
            trainer.test(self.model, test_datamodule)

    def setup_compute_devices(self):
        cuda_gpus = torch.cuda.device_count()
        mps_available = torch.backends.mps.is_available() and not torch.cuda.is_available()

        if cuda_gpus:
            self.trainer_args.update(dict(precision="16-mixed"))
            if cuda_gpus == 1:
                print("Single GPU ============================")
                self.trainer_args.update(dict(accelerator="gpu", devices=1))
            else:
                print("Multiple GPU ============================")
                self.trainer_args.update(
                    dict(
                        accelerator="gpu",
                        devices=cuda_gpus,
                        strategy=DDPStrategy(find_unused_parameters=True),
                    )
                )
        elif mps_available:
            # Apple-silicon Metal Performance Shaders backend
            self.trainer_args.update(dict(accelerator="mps", devices=1, precision="16-mixed"))
        else:
            self.trainer_args.update(dict(accelerator="cpu", devices=1, precision=32))

