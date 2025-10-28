"""
Training script for BERTweet classification using PyTorch Lightning.
- Uses TensorBoard for metrics logging (and optionally CSV).
- Reads a single CSV with columns: text, label


Example:
python train_bertweet.py \
--data_csv ../tweets_data/train_master.csv \
--num_labels 3 \
--text_col text --label_col AR \
--val_size 0.15 --test_size 0.15 \
--batch_size 64 \
--lr 2e-5 \
--max_epochs 6 \
--class_weight


Launch TensorBoard:
tensorboard --logdir outputs/tb_logs
"""

import os
import json
import datetime
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import pandas as pd

from modules.berttweet import TweetsTVTDataModule, BertweetModule

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_csv', type=str, required=True)
    p.add_argument('--text_col', type=str, default='text')
    p.add_argument('--label_col', type=str, default='AR')
    p.add_argument('--num_labels', type=int, required=True)
    p.add_argument('--val_size', type=float, default=0.2)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--test_seed', type=int, default=2025)
    p.add_argument('--val_seed', type=int, default=2025)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=0.1)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--max_epochs', type=int, default=6)
    p.add_argument('--precision', type=str, default='bf16', choices=['32', '16', 'bf16'])
    p.add_argument('--accumulate_grad_batches', type=int, default=1)
    p.add_argument('--class_weight', action='store_true')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--output_dir', type=str, default='outputs/logs')
    return p.parse_args()

def main():
    args = parse_args()
    pl.seed_everything(args.test_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    dm = TweetsTVTDataModule(
        data_path=args.data_csv,
        text_col=args.text_col,
        label_col=args.label_col,
        label_names=["Problem", "Solution", "Other"],
        num_labels=args.num_labels,
        val_size=args.val_size,
        test_size=args.test_size,
        val_seed=args.val_seed,
        test_seed=args.test_seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )
    dm.setup()

    # Model
    model = BertweetModule(
        num_labels=args.num_labels,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        class_weight=args.class_weight,
        train_label_counts=dm.train_label_counts,
    )
    
    output_dir = os.path.join(
        args.output_dir, 
        datetime.datetime.now().strftime("%Y%m%d_%H%M")
        )

    # Loggers: TensorBoard (always), CSV (optional)
    tb_logger = TensorBoardLogger(save_dir=output_dir, name='tb')
    loggers = [tb_logger]

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "ckpt"),
        filename='bertweet-{epoch:02d}-{val_macro_f1:.4f}',
        monitor='val_macro_f1',
        mode='max',
        save_top_k=1,
        save_last=True,
    )
    early_cb = EarlyStopping(monitor='val_macro_f1', mode='max', patience=3)
    lr_cb = LearningRateMonitor(logging_interval='step')

    precision_map = {'32': 32, '16': '16-mixed', 'bf16': 'bf16-mixed'}

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        max_epochs=args.max_epochs,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=precision_map.get(args.precision, 'bf16-mixed'),
        callbacks=[ckpt_cb, early_cb, lr_cb],
        logger=loggers,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)
    print(f"Done training. Best checkpoint: {ckpt_cb.best_model_path}")
    
    # Evaluate on test set (only if test_size > 0)
    if args.test_size > 0:
        test_metrics = trainer.test(ckpt_path=ckpt_cb.best_model_path, datamodule=dm)
        print("Test metrics:", {k: float(v) if hasattr(v, "item") else v for k, v in test_metrics[0].items()})
    else:
        print("No internal test split (test_size=0.0) - using external test set for evaluation")
        test_metrics = None
    
    # Log hyperparameters to TensorBoard
    hparams = {
        'model_name': 'vinai/bertweet-base',
        'num_labels': args.num_labels,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'val_size:': args.val_size,
        'test_size:': args.test_size,
        'warmup_ratio': args.warmup_ratio,
        'batch_size': args.batch_size,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'max_length': args.max_length,
        'precision': args.precision,
        'class_weight': bool(args.class_weight),
        'val_seed': args.val_seed,
        'test_seed': args.test_seed,
        }
    text = "```json\n" + json.dumps(hparams, indent=2, sort_keys=True) + "\n```"
    tb_logger.experiment.add_text("hparams", text, global_step=trainer.global_step)
    
    if test_metrics is not None:
        text = "```json\n" + json.dumps(test_metrics[0], indent=2, sort_keys=True) + "\n```"
        tb_logger.experiment.add_text("test_metrics", text, global_step=trainer.global_step)
    
    #### Write test set predictions ####
    # Get predicted logits for each example in the test set (only if test_size > 0)
    if args.test_size > 0:
        pred_batches = trainer.predict(model=model, dataloaders=dm.test_dataloader(), ckpt_path=ckpt_cb.best_model_path)

        # Concatenate logits in original row order
        logits = torch.cat([b["logits"] for b in pred_batches], dim=0)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        pred_classes = probs.argmax(axis=1) + 1  # Add 1 to reflect original labelling codes

        # Pull original texts and labels (if available) from the test dataset
        test_tweets = getattr(dm.test_ds, "texts", None)
        test_labels = getattr(dm.test_ds, "labels", None)
        test_labels = [t + 1 for t in test_labels]  # Add 1 to reflect original labelling codes

        # Build a nice dataframe
        out = {
            "tweet": test_tweets if test_tweets is not None else [None] * len(pred_classes),
            "pred_label": pred_classes,
            "pred_confidence": probs.max(axis=1),          # highest class prob
            "probs": [probs[i].tolist() for i in range(len(pred_classes))],
        }
        if test_labels is not None:
            out["label"] = test_labels

        df_pred = pd.DataFrame(out)
        csv_path = f"{output_dir}/test_predictions.csv"
        df_pred.to_csv(csv_path, index=False)
    else:
        print("Skipping internal test predictions (test_size=0.0)")
        

if __name__ == '__main__':
    main()  