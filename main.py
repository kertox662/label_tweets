import argparse
import time
import yaml
import torch
import warnings

from config import SupervisedTrainingConfig
from training import AutomodelSupervisedTrainer

def run_classification(args, config):
    trainer = AutomodelSupervisedTrainer(config)
    trainer.train(
        use_full_test_data=args.test_full_test_set,
        label_all_tweets=args.label_tweets,
        folds = args.folds
    )

def run_hyperparameter(config):
    print("Hyperparameter search not currently converted.")

def main():
    parser = argparse.ArgumentParser(description="Model training interface")
    parser.add_argument('--config', required=True, help="Path to the config file")
    parser.add_argument('--seed', type=int, default=time.time(), help="Random seed for reproducibility")
    parser.add_argument("--test-full-test-set", action="store_true")
    parser.add_argument("--label-tweets", action="store_true")
    parser.add_argument("--folds", type=int, default=None)

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)

    # Dispatch
    run_classification(args, config)

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_float32_matmul_precision('medium')
    main()