import argparse
import yaml
import torch

from classification.config import SupervisedTrainingConfig
from classification.training import SupervisedTrainer
from self_supervised.config import SelfSupervisedTrainingConfig
from self_supervised.training import SelfSupervisedTrainer

def run_ssl(config: SelfSupervisedTrainingConfig):
    trainer = SelfSupervisedTrainer(config)
    out = trainer.train()
    print(out)

def run_classification(args, config):
    trainer = SupervisedTrainer(config)
    trainer.train(args.test_full_test_set)

def run_hyperparameter(config):
    print("Hyperparameter search not currently converted.")

def main():
    parser = argparse.ArgumentParser(description="Model training interface")
    parser.add_argument('--config', required=True, help="Path to the config file")
    cmd_parser = parser.add_subparsers(dest="command", required=True, help="Task to run")

    cmd_parser.add_parser("ssl")
    class_parser = cmd_parser.add_parser("classification")
    class_parser.add_argument("--test-full-test-set", action="store_true")
    cmd_parser.add_parser("hyperparameter")

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)

    # Dispatch
    if args.command == "ssl":
        config = SelfSupervisedTrainingConfig(**config_yaml)
        run_ssl(config)
    elif args.command == "classification":
        config = SupervisedTrainingConfig(**config_yaml)
        run_classification(args, config)
    elif args.command == "hyperparameter":
        run_hyperparameter(args.config)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()