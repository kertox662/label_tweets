# Configuration constants for the tweet classification project
import torch.nn as nn
# Training parameters
EPOCHS = 25
EARLY_STOPPING_EPOCH = 5
EARLY_STOPPING_MIN_DELTA = 0.0001
DEFAULT_TRIALS_PER_PARAMS = 1

# File paths
SUMMARY_FILE = "search_summary.csv"
PARAM_SEARCH_LOG_NAME = "hp_search"

# Data module parameters
datamodule_params = {
    "batch_size": 64,
    "target_col": 'AR',
    "test_size": 0.2,
    "validation_size": 0.2,
    "oversample": True,
    "random_state": 2025
}

base_datamodule_params = {
    "target_col": 'AR',
    "test_size": 0.2,
    "validation_size": 0.2,
    "random_state": 2025
}


import os

# Model options for hyperparameter search
MODEL_OPTIONS = [
    "sentence-transformer-trained-tweet"
    #  "pre_trained_sentence_st" if os.path.exists("pre_trained_sentence_st") else "sentence-transformers/all-mpnet-base-v2"
    # "sentence-transformers/all-mpnet-base-v2",
    # "vinai/bertweet-base",
]

SOFT_LABEL_OPTIONS = [True, False]
LEARNING_RATE_OPTIONS = [1e-4, 1e-3, 5e-3, 1e-2]
FREEZE_ENCODER_OPTIONS = [True]
BATCH_SIZE_OPTIONS = [64]
OVERSAMPLE_OPTIONS = [False]
CLASS_WEIGHT_OPTIONS = [True, False]

# Hidden layer options (commented out as they're not used in current search)
HIDDEN_DIM_OPTIONS = [64]
DROPOUT_OPTIONS = [0.25, 0.5, 0.75]
ACTIVATION_FUNCTION_OPTIONS = [nn.ReLU()] 