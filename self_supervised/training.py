import os
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from self_supervised.config import SelfSupervisedTrainingConfig
from self_supervised.model import TransformerBackbone
from data.unsupervised import TweetsDataModuleUnsupervised

class SelfSupervisedTrainer:
    def __init__(self, config: SelfSupervisedTrainingConfig):
        self.config = config

        if config.model_name:
            self.model = TransformerBackbone(
                model_name=config.model_name,
                learning_rate=config.learning_rate
            )
        elif config.checkpoint_name:
            if not os.path.isfile(config.checkpoint_name):
                print(f"No checkpoint found at: {config.checkpoint_name:}")
                raise FileNotFoundError(config.checkpoint_name)
            print(f"Loading model from checkpoint: {config.checkpoint_name}")
            self.model = TransformerBackbone.load_from_checkpoint(config.checkpoint_name)
        else:
            raise ValueError("Must have model name or checkpoint name.")
        
        self.early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.stopping_patience,
        )
        self.checkpoint = ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            save_last= True,
            filename="st-ssl-{epoch:02d}-{val_loss:.3f}",
            save_top_k=1,
            mode="min",
            every_n_epochs=config.checkpoint_epochs,
        )

        self.trainer_args = dict(
            max_epochs=config.max_epochs,
            val_check_interval = config.validation_interval,
            callbacks=[self.early_stopping, self.checkpoint],
            logger=TensorBoardLogger(config.logs_dir, name=config.logs_name),
            log_every_n_steps=config.logging_interval_steps,
            accumulate_grad_batches=config.accumulate_grad_batches,
        )
    
    def train(self):
        self.setup_compute_devices()
        trainer = pl.Trainer(**self.trainer_args)
        
        data = pd.read_csv(self.config.data_path)
        datamodule = TweetsDataModuleUnsupervised(data=data, batch_size=self.config.batch_size, num_workers=1)
        print("Data Read ========================================")
        
        print("Trainer Fitting ========================================")
        trainer.fit(self.model, datamodule)

        # Save the fine-tuned encoder for downstream tasks
        output_dir = self.model.save_encoder(self.config.save_path)
        print(f"Self-supervised model saved to: {output_dir}")
        return output_dir

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
