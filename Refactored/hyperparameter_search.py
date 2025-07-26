import itertools
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

from .config import (
    EPOCHS, EARLY_STOPPING_EPOCH, EARLY_STOPPING_MIN_DELTA, DEFAULT_TRIALS_PER_PARAMS,
    PARAM_SEARCH_LOG_NAME, base_datamodule_params,
    MODEL_OPTIONS, SOFT_LABEL_OPTIONS, LEARNING_RATE_OPTIONS, FREEZE_ENCODER_OPTIONS,
    BATCH_SIZE_OPTIONS, OVERSAMPLE_OPTIONS, CLASS_WEIGHT_OPTIONS
)
from .model import BertweetClassifier
from .data_module import TweetsDataModule


def evaluate_with_params(dataModule, model_params):
    model = BertweetClassifier(**model_params)
    model.id2label = dataModule.id2label          # list   -> e.g. ["neg","neu","pos"]
    model.label2id = dataModule.label2id
    logger = TensorBoardLogger("tb_logs", name=PARAM_SEARCH_LOG_NAME)

    early_stop_callback = EarlyStopping(
        monitor="val_f1_ar", patience=EARLY_STOPPING_EPOCH, min_delta=EARLY_STOPPING_MIN_DELTA,
        verbose=True, mode="max")
    trainer = pl.Trainer(max_epochs=EPOCHS, logger=logger, enable_checkpointing=False, callbacks=[early_stop_callback])
    trainer.fit(model=model, datamodule=dataModule)

    return trainer.test(model=model, datamodule=dataModule)      # <- returns a list of dicts


def param_search(param_iterator, trials_per_param=DEFAULT_TRIALS_PER_PARAMS, summary_file="hp_search_summary.csv"):
    result = []
    for (model_params, data_params) in param_iterator:
        dataModule = TweetsDataModule.read_csv(**data_params)
        dataModule.setup("fit")

        if model_params["class_weights"] is not None:
            model_params["class_weights"] = dataModule.train_class_weights()

        for _ in range(trials_per_param):
            trial = evaluate_with_params(dataModule, model_params)
            results = trial[0]
            trial_results = {
                **model_params,
                **results
            }
            result.append(trial_results)
        results_df = pd.DataFrame(result)
        results_df.to_csv(summary_file)


def create_model_options(model_name, learning_rate, freeze_encoder, classifier_constructor, class_weights, soft_label):
    return {
        "transformer_model_name": model_name,
        "learning_rate": learning_rate,
        "freeze_encoder": freeze_encoder,
        "classifier_constructor": classifier_constructor,
        "class_weights": class_weights,
        "use_soft_labels": soft_label,
    }


def create_data_options(batch_size, oversample):
    return {
        "batch_size": batch_size,
        "oversample": oversample,
        **base_datamodule_params,
    }


def create_linear_param_iterator():
    """Do all linear layer options"""
    print(f"MODEL_OPTIONS: {MODEL_OPTIONS}")
    print(f"BATCH_SIZE_OPTIONS: {BATCH_SIZE_OPTIONS}")
    print(f"OVERSAMPLE_OPTIONS: {OVERSAMPLE_OPTIONS}")
    print(f"LEARNING_RATE_OPTIONS: {LEARNING_RATE_OPTIONS}")
    print(f"CLASS_WEIGHT_OPTIONS: {CLASS_WEIGHT_OPTIONS}")
    
    for (model, batch, soft_label, oversample, freeze, lr, weight) in itertools.product(
        MODEL_OPTIONS, BATCH_SIZE_OPTIONS, SOFT_LABEL_OPTIONS,
        OVERSAMPLE_OPTIONS, FREEZE_ENCODER_OPTIONS,
        LEARNING_RATE_OPTIONS, CLASS_WEIGHT_OPTIONS,
    ):
        print("----------------------------------------------------------------------------------------------------\n")
        print(f"batch: {batch}, oversample: {oversample}")
        print(f"freeze: {freeze}, lr: {lr}, class_weight: {weight}")
        # Skip over oversampling and class weights being done at the same time.
        if oversample and weight is not None:
            continue

        yield create_model_options(model, lr, freeze, None, weight, soft_label), create_data_options(batch, oversample)


def create_one_hidden_layer_param_iterator():
    """Do all 1-hidden layer options"""
    # Note: This requires HIDDEN_DIM_OPTIONS, DROPOUT_OPTIONS, ACTIVATION_FUNCTION_OPTIONS to be defined
    # These are currently commented out in config.py
    pass


def create_two_hidden_layer_param_iterator():
    """Do all 2-hidden layer options"""
    # Note: This requires HIDDEN_DIM_OPTIONS, DROPOUT_OPTIONS, ACTIVATION_FUNCTION_OPTIONS to be defined
    # These are currently commented out in config.py
    pass 