import copy
import time
import os

import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from classification.config import SupervisedTrainingConfig
from data.supervised import TweetsDataModule
from modules.berttweet import BertweetModule, TweetsTVTDataModule, create_k_fold_data_modules

TEXT_COL = "text"
LABEL_COL = "AR"
LABEL_NAMES = ["Problem", "Solution", "Other"]
NUM_LABELS = 3
MAX_LENGTH = 128

class AutomodelSupervisedTrainer:
    def __init__(self, config: SupervisedTrainingConfig):
        self.config = config

        
        self.model = None
        self.model_config = {
            "model_name": config.model_name,
            "num_labels": NUM_LABELS,
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "dropout": config.dropout_p,
            "class_weight": config.class_weight,
        }
        
        self.early_stopping = EarlyStopping(
            monitor='val_macro_f1', mode='max',
            patience=config.stopping_patience,
        )
        self.ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join("outputs", "ckpt"),
        filename='bertweet-{epoch:02d}-{val_macro_f1:.4f}',
        monitor='val_macro_f1',
        mode='max',
        save_top_k=1,
        save_last=True,
    )

        self.trainer_args = dict(
            max_epochs=config.max_epochs,
            callbacks=[self.early_stopping, self.ckpt_cb],
            logger=TensorBoardLogger(config.logs_dir, name=config.logs_name),
            accumulate_grad_batches=config.accumulate_grad_batches,
            deterministic=True,
        )
    
    def train(
            self,
            use_full_test_data=False,
            label_all_tweets=False,
            folds = None):
        pl.seed_everything(self.config.global_seed)
        self.setup_compute_devices()

        dm_config = {
            "text_col": TEXT_COL,
            "label_col": LABEL_COL,
            "label_names": LABEL_NAMES,
            "num_labels": NUM_LABELS,
            "model_name": self.config.model_name,
            "batch_size": self.config.batch_size,
            "max_length": MAX_LENGTH,
            "num_workers": 12,
            "val_seed": self.config.val_data_seed,
            "test_seed": self.config.test_data_seed
        }

        if folds is not None:
            k_fold_data_modules = create_k_fold_data_modules(
                data_path=self.config.kfold_data,
                num_folds=folds,
                **dm_config
            )

            metrics_acc = {}

            for fold_idx, dm in enumerate(k_fold_data_modules):
                print(f"Fold {fold_idx + 1} Training ========================================")
                dm.setup()
                if self.model is None:
                    self.model = BertweetModule(
                        **self.model_config,
                        train_label_counts=dm.train_label_counts,
                        tokenizer=dm.tokenizer,
                    )
                # Start with a fresh model for each fold
                model_copy = copy.deepcopy(self.model)
                trainer = pl.Trainer(**self.trainer_args)
                trainer.fit(model_copy, dm)
                print(f"Fold {fold_idx + 1} Testing ========================================")
                metrics = trainer.test(model_copy, dm)
                self.add_to_test_metric_accumulator(metrics_acc, metrics)
            self.model = model_copy

            avg_metrics = {
                f"avg_{k}": sum(v)/folds for k,v in metrics_acc
            }

            self.model.log_dict(avg_metrics)
            
            print("=============== K-Fold metrics ===============")
            for k,v in avg_metrics:
                print(f"{k}:\t{v}")

            input("Done cross validation.\nPress <Enter> to continue...")

        dm = TweetsTVTDataModule(
                data_path=self.config.train_data,
                **dm_config
            )
        dm.setup()

        self.model = BertweetModule(
            **self.model_config,
            train_label_counts=dm.train_label_counts,
            tokenizer=dm.tokenizer,
        )

        trainer = pl.Trainer(**self.trainer_args)
    
        print("Data Read ========================================")
        
        print("Trainer Fitting ========================================")
        trainer.fit(self.model, dm)

        print(f"Done training. Best checkpoint: {self.ckpt_cb.best_model_path}")


        print("Trainer Testing ========================================")
        trainer.test(ckpt_path=self.ckpt_cb.best_model_path, datamodule=dm)

        if use_full_test_data:
            input("Press <Enter> to continue...")
            print("Full Test Set ================")
            dm_test = TweetsTVTDataModule(
                data_path=self.config.test_data,
                **dm_config,
                test_only=True,
            )
            dm_test.setup()
            trainer.test(ckpt_path=self.ckpt_cb.best_model_path, datamodule=dm_test)

        if label_all_tweets:
            input("Press <Enter> to continue...")
            print("Labeling all tweets ================")
            # Load all tweets data
            all_tweets_df = pd.read_csv(self.config.all_data)
            
            # Create a prediction-only data module
            dm_all = TweetsTVTDataModule(
                data_path=self.config.all_data,
                text_col=TEXT_COL,
                label_col=LABEL_COL,
                label_names=LABEL_NAMES,
                num_labels=NUM_LABELS,
                batch_size=self.config.batch_size,
                max_length=MAX_LENGTH,
                num_workers=12,
                test_only=True,
            )
            dm_all.setup()
            
            # self.trainer_args.update(dict(accelerator="cpu", devices=1, precision=32))

            # Get predictions using existing trainer configuration
            predict_trainer = pl.Trainer(**self.trainer_args)
            predictions = predict_trainer.predict(ckpt_path=self.ckpt_cb.best_model_path, datamodule=dm_all, model=self.model)
            
            # Process predictions - extract logits from dictionaries
            all_logits = torch.cat([batch["logits"] for batch in predictions], dim=0)
            all_probs = torch.softmax(all_logits, dim=1)
            all_labels = torch.argmax(all_probs, dim=1)
            
            # Create output dataframe
            output_df = pd.DataFrame({
                'tweet_id': all_tweets_df.index if 'tweet_id' not in all_tweets_df.columns else all_tweets_df['tweet_id'],
                'label': [LABEL_NAMES[label.item()] for label in all_labels],
            })
            
            # Add confidence columns for each class
            for i, label_name in enumerate(LABEL_NAMES):
                output_df[f'confidence_{label_name}'] = all_probs[:, i].cpu().numpy()
            
            # Save to file
            output_path = f"{self.config.logs_dir}/{self.config.logs_name}_{int(time.time())}_predictions.csv"
            output_df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")

    def setup_compute_devices(self):
        cuda_gpus = torch.cuda.device_count()
        mps_available = torch.backends.mps.is_available() and not torch.cuda.is_available()

        if cuda_gpus:
            self.trainer_args.update(dict(precision="bf16-mixed"))
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

    def add_to_test_metric_accumulator(self, acc, metrics):
        for k,v in metrics:
            if k not in acc:
                acc[k] = []
            acc[k].append(v)
        return acc