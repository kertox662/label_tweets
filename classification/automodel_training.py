import copy
import time
import os
import json
from typing import Generator, Tuple

import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from classification.config import SupervisedTrainingConfig
from modules.berttweet import BertweetModule, TweetsTVTDataModule, create_k_fold_data_modules, KFoldDataModule

TEXT_COL = "text"
LABEL_COL = "AR"
LABEL_NAMES = ["Problem", "Solution", "Other"]
NUM_LABELS = 3
MAX_LENGTH = 128
SEPARATOR = "========================================"

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

        self.dm_config = {
            "text_col": TEXT_COL,
            "label_names": LABEL_NAMES,
            "num_labels": NUM_LABELS,
            "model_name": self.config.model_name,
            "batch_size": self.config.batch_size,
            "max_length": MAX_LENGTH,
            "num_workers": 12,
            "val_size": self.config.val_size,
        }

        self.trainer_args = dict(
            max_epochs=config.max_epochs,
            logger=TensorBoardLogger(config.logs_dir, name=config.logs_name),
            gradient_clip_val=1.0,
            accumulate_grad_batches=config.accumulate_grad_batches,
            deterministic=True,
            enable_model_summary=False,
            log_every_n_steps=10,
        )

        self.gold_test_primary_label = TweetsTVTDataModule(
            data_path=self.config.test_data,
            **self.dm_config,
            test_only=True,
            label_col=LABEL_COL
        )
        self.gold_test_primary_label.setup()

        self.gold_test_opposite_label = TweetsTVTDataModule(
            data_path=self.config.test_data,
            **self.dm_config,
            test_only=True,
            label_col=self.opposite_label()
        )
        self.gold_test_opposite_label.setup()

        self.dm_config.update(dict(label_col=LABEL_COL))

    def create_callbacks(self):
        self.early_stopping = EarlyStopping(
            monitor='val_macro_f1', mode='max',
            patience=self.config.stopping_patience,
        )
        self.ckpt_cb = ModelCheckpoint(
            dirpath=os.path.join("outputs", "ckpt"),
            filename='bertweet-{epoch:02d}-{val_macro_f1:.4f}',
            monitor='val_macro_f1',
            mode='max',
            save_top_k=1,
            save_last=False,
        )
        return [self.early_stopping, self.ckpt_cb]

    def opposite_label(self):
        return "MB" if LABEL_COL == "AR" else "AR"
 
    def train(
            self,
            use_full_test_data=False,
            label_all_tweets=False,
            folds = None):
        self.setup_compute_devices()
        self.timestamp = int(time.time())

        if self.config.cross_val_folds is not None:
            crossval_df = self.run_k_times(
                self.config.trials,
                self.cross_validation,
                self.cross_validation_datamodule,
                label = "Cross Validation Trial {}"
            )

            crossval_path = f"{self.config.logs_dir}/{self.config.logs_name}_{self.timestamp}_crossval.csv"
            crossval_df.to_csv(crossval_path)
    
        training_df = self.run_k_times(
            self.config.trials,
            self.training_trial,
            self.training_trial_datamodule,
            label = "Training Trial {}",
            remove_disagreements = True,
            use_full_test_data=use_full_test_data,
        )
        training_results_path = f"{self.config.logs_dir}/{self.config.logs_name}_{self.timestamp}_training_agreed.csv"
        training_df.to_csv(training_results_path)

        training_diagree_df = self.run_k_times(
            self.config.trials,
            self.training_trial,
            self.training_trial_datamodule,
            label = "Training Disagree Trial {}",
            remove_disagreements = False,
            use_full_test_data=use_full_test_data,
        )
        training_results_path = f"{self.config.logs_dir}/{self.config.logs_name}_{self.timestamp}_training_disagreed.csv"
        training_diagree_df.to_csv(training_results_path)

        pl.seed_everything(self.config.global_seed)

        dm = TweetsTVTDataModule(
                data_path=self.config.train_with_test_data,
                **self.dm_config,
                val_seed = self.config.val_data_seed,
                test_seed = self.config.test_data_seed,
                no_test=True,
            )
        dm.setup()

        self.model = BertweetModule(
            **self.model_config,
            train_label_counts=dm.train_label_counts,
        )
        trainer = pl.Trainer(**self.trainer_args, callbacks=self.create_callbacks())

        print("Trainer Fitting ========================================")
        trainer.fit(self.model, dm)

        print(f"Done training. Best checkpoint: {self.ckpt_cb.best_model_path}")

        if label_all_tweets:
            if "strategy" in self.trainer_args:
                self.trainer_args["strategy"] = DDPStrategy(find_unused_parameters=False, accelerator="gpu")
            
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

            # Get predictions using existing trainer configuration
            predict_trainer = pl.Trainer(**self.trainer_args)
            predictions = predict_trainer.predict(ckpt_path=self.ckpt_cb.best_model_path, datamodule=dm_all, model=self.model)
            
            # Process predictions - extract logits from dictionaries
            all_logits = torch.cat([batch["logits"] for batch in predictions], dim=0)
            all_probs = torch.softmax(all_logits, dim=1).float()
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
            output_path = f"{self.config.logs_dir}/{self.config.logs_name}_{self.timestamp}_predictions.csv"
            output_df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")

    def run_k_times(self, k, training_function, datamodule_factory, label = "Iteration {} ", **kwargs):
        pl.seed_everything(self.config.global_seed)

        metrics_df = pd.DataFrame()
        for i in range(k):
            if "strategy" in self.trainer_args:
                self.trainer_args["strategy"] = DDPStrategy(find_unused_parameters=False, accelerator="gpu")
            print(f"{label.format(i + 1)} {SEPARATOR}")
            dm = datamodule_factory(i, **kwargs)
            results = training_function(i, dm)
            metrics_df = pd.concat([metrics_df, results], ignore_index=True)
            
            # Clean up after each iteration
            torch.cuda.empty_cache()
            
        return metrics_df

    def create_metrics_dict(self):
        return {
            "iteration": [],
            "test_loss": [],
            "test_acc": [],
            "test_macro_f1": [],
            "test_weighted_f1": [],
            "epochs": [],
            "cm": [],
            "train_size": [],
            "val_size": [],
            "test_size": [],
        }

    def cross_validation_datamodule(self, iteration: int):
        return create_k_fold_data_modules(
            data_path=self.config.train_with_test_data,
            num_folds=self.config.cross_val_folds,
            **self.dm_config,
            remove_disagreements=True,
        )

    def cross_validation(self, iteration: int, k_fold_datamodule_generator):
        metrics_acc = {
            **self.create_metrics_dict(),
            "fold": []
        }

        self.model = None

        for fold_idx, dm in k_fold_datamodule_generator:
            print(f"It {iteration + 1}, Fold {fold_idx + 1} Training {SEPARATOR}")
            dm.setup()
            if self.model is None:
                self.model = BertweetModule(
                    **self.model_config,
                    train_label_counts=dm.train_label_counts,
                )
            
            # Start with a fresh model for each fold
            model_copy = copy.deepcopy(self.model)
            trainer = pl.Trainer(**self.trainer_args, callbacks=self.create_callbacks())
            trainer.fit(model_copy, dm)

            # Get the metrics for the fold
            print(f"It {iteration + 1}, Fold {fold_idx + 1} Testing {SEPARATOR}")
            metrics = trainer.test(ckpt_path=self.ckpt_cb.best_model_path, datamodule=dm)[0] # From testing
            
            # Add additional metrics to the dict
            metrics["iteration"] = iteration
            metrics["fold"] = fold_idx
            metrics["epochs"] = trainer.current_epoch
            metrics["cm"] = str(model_copy.cm.compute().tolist())
            metrics["train_size"] = len(dm.train_ds)
            metrics["val_size"] = len(dm.val_ds)
            metrics["test_size"] = len(dm.test_ds)

            # Merge with previous folds.
            self.add_to_test_metric_accumulator(metrics_acc, metrics)
            if os.path.exists(self.ckpt_cb.best_model_path):
                os.remove(self.ckpt_cb.best_model_path)
            del model_copy
            torch.cuda.empty_cache()
        
        return pd.DataFrame(metrics_acc)

    def training_trial_datamodule(self, iteration, use_full_test_data: bool = False, remove_disagreements: bool = False):
        if use_full_test_data:
            dm = TweetsTVTDataModule(
                data_path=self.config.train_data,
                **self.dm_config,
                val_seed = self.config.val_data_seed + iteration,
                test_seed = self.config.test_data_seed + iteration,
                no_test=True,
                remove_disagreements=remove_disagreements
            )
            dm.setup()

            return {
                "train": dm,
                "test_primary": self.gold_test_primary_label,
                "test_opposite": self.gold_test_opposite_label
            }

        else:
            dm = TweetsTVTDataModule(
                data_path=self.config.train_data,
                **self.dm_config,
                val_seed = self.config.val_data_seed + iteration,
                test_seed = self.config.test_data_seed + iteration,
                remove_disagreements=remove_disagreements
            )
            dm.setup()

            self.dm_config["label_col"] = self.opposite_label()
            opp_dm = TweetsTVTDataModule(
                data_path=self.config.train_data,
                **self.dm_config,
                val_seed = self.config.val_data_seed + iteration,
                test_seed = self.config.test_data_seed + iteration,
                remove_disagreements=remove_disagreements
            )
            opp_dm.setup()
            self.dm_config["label_col"] = LABEL_COL

            return {
                "train": dm,
                "test_primary": dm,
                "test_opposite": opp_dm
            }


    def training_trial(self,
                       iteration: int,
                       datamodule: TweetsTVTDataModule,
                       use_full_test_data: bool = False,
                       **kwargs
                    ):
        metrics_acc = {
            **self.create_metrics_dict(),
            "label": []
        }

        # Create and fit the model
        self.model = BertweetModule(
            **self.model_config,
            train_label_counts=datamodule["train"].train_label_counts,
        )

        print(self.trainer_args, flush=True)
        trainer = pl.Trainer(**self.trainer_args, callbacks=self.create_callbacks())
        trainer.fit(self.model, datamodule=datamodule["train"])

        main_metrics = trainer.test(ckpt_path=self.ckpt_cb.best_model_path, datamodule=datamodule["test_primary"])[0]
        main_metrics["iteration"] = iteration
        main_metrics["label"] = LABEL_COL
        main_metrics["epochs"] = trainer.current_epoch
        main_metrics["cm"] = str(self.model.cm.compute().tolist())
        main_metrics["train_size"] = len(datamodule["train"].train_ds)
        main_metrics["val_size"] = len(datamodule["train"].val_ds)
        main_metrics["test_size"] = len(datamodule["test_primary"].test_ds)
        self.add_to_test_metric_accumulator(metrics_acc, main_metrics)

        opp_metrics = trainer.test(ckpt_path=self.ckpt_cb.best_model_path, datamodule=datamodule["test_opposite"])[0]
        opp_metrics["iteration"] = iteration
        opp_metrics["label"] = self.opposite_label()
        opp_metrics["epochs"] = trainer.current_epoch
        opp_metrics["cm"] = str(self.model.cm.compute().tolist())
        opp_metrics["train_size"] = len(datamodule["train"].train_ds)
        opp_metrics["val_size"] = len(datamodule["train"].val_ds)
        opp_metrics["test_size"] = len(datamodule["test_primary"].test_ds)
        self.add_to_test_metric_accumulator(metrics_acc, opp_metrics)
        
        if os.path.exists(self.ckpt_cb.best_model_path):
            os.remove(self.ckpt_cb.best_model_path)

        del self.model
        torch.cuda.empty_cache()
        self.model = None
        return pd.DataFrame(metrics_acc)

    def setup_compute_devices(self):
        cuda_gpus = torch.cuda.device_count()
        mps_available = torch.backends.mps.is_available() and not torch.cuda.is_available()

        if cuda_gpus:
            self.trainer_args.update(dict(precision="bf16-mixed", devices=cuda_gpus))
            if cuda_gpus == 1:
                print("Single GPU ============================")
                self.trainer_args.update(dict(accelerator="gpu"))
            else:
                print("Multiple GPU ============================")
                # When using DDPStrategy, don't set accelerator - let the strategy handle it
                self.trainer_args.update(
                    dict(
                        accelerator="auto",
                        strategy=DDPStrategy(find_unused_parameters=False, accelerator="gpu"),
                    )
                )
        elif mps_available:
            # Apple-silicon Metal Performance Shaders backend
            self.trainer_args.update(dict(accelerator="mps", devices=1, precision="16-mixed"))
        else:
            self.trainer_args.update(dict(accelerator="cpu", devices=1, precision=32))

    def add_to_test_metric_accumulator(self, acc, metrics):
        for k,v in metrics.items():
            if k not in acc:
                acc[k] = []
            acc[k].append(v)
        return acc