import itertools
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import torch

from .config import (
    EPOCHS, EARLY_STOPPING_EPOCH, EARLY_STOPPING_MIN_DELTA, DEFAULT_TRIALS_PER_PARAMS,
    PARAM_SEARCH_LOG_NAME, base_datamodule_params,
    MODEL_OPTIONS, SOFT_LABEL_OPTIONS, LEARNING_RATE_OPTIONS, FREEZE_ENCODER_OPTIONS,
    BATCH_SIZE_OPTIONS, OVERSAMPLE_OPTIONS, CLASS_WEIGHT_OPTIONS,
    HIDDEN_DIM_OPTIONS,
    DROPOUT_OPTIONS,
    ACTIVATION_FUNCTION_OPTIONS,
)
from .model import BertweetClassifier
from .data_module import TweetsDataModule


def evaluate_with_params(dataModule, model_params):
    # Remove keys that are not accepted by BertweetClassifier
    model_hparams = {k: v for k, v in model_params.items() if k not in {"hidden_dim", "activation"}}
    model = BertweetClassifier(**model_hparams)
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

        # ------------------------------------------------------------------
        # Handle class weights: compute tensor if requested (bool or not None)
        # ------------------------------------------------------------------
        if model_params["class_weights"] is True:   # explicit request
            model_params["class_weights"] = dataModule.train_class_weights()
        else:
            model_params["class_weights"] = None

        for _ in range(trials_per_param):
            trial = evaluate_with_params(dataModule, model_params)
            results = trial[0]
            # --------------------------------------------------------------
            # Build a *sanitised* param dictionary for logging to CSV.
            # Keep only simple scalars; drop objects / tensors.
            # --------------------------------------------------------------
            clean_params = {
                "transformer_model_name": model_params["transformer_model_name"],
                "learning_rate": model_params["learning_rate"],
                "freeze_encoder": model_params["freeze_encoder"],
                "use_soft_labels": model_params["use_soft_labels"],

                # Wanted by the user:
                "batch_size": data_params.get("batch_size"),
                "hidden_dim": model_params.get("hidden_dim"),
                "activation": model_params.get("activation"),
                "use_class_weights": model_params["class_weights"] is not None,
                "class_weights": (
                    ",".join(map(lambda x: f"{x:.4f}", model_params["class_weights"].tolist()))
                    if isinstance(model_params["class_weights"], torch.Tensor)
                    else ""
                ),
            }

            trial_results = {**clean_params, **results}
            result.append(trial_results)
        results_df = pd.DataFrame(result)
        results_df.to_csv(summary_file, index=False)


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
    """Yield all parameter combinations that use a 1-hidden-layer classifier."""
    from .classifier_constructors import one_layer_classifier_constructor

    # Print chosen search spaces for visibility
    print(f"MODEL_OPTIONS: {MODEL_OPTIONS}")
    print(f"BATCH_SIZE_OPTIONS: {BATCH_SIZE_OPTIONS}")
    print(f"OVERSAMPLE_OPTIONS: {OVERSAMPLE_OPTIONS}")
    print(f"LEARNING_RATE_OPTIONS: {LEARNING_RATE_OPTIONS}")
    print(f"CLASS_WEIGHT_OPTIONS: {CLASS_WEIGHT_OPTIONS}")
    print(f"HIDDEN_DIM_OPTIONS: {HIDDEN_DIM_OPTIONS}")
    print(f"DROPOUT_OPTIONS: {DROPOUT_OPTIONS}")
    print(f"ACTIVATION_FUNCTION_OPTIONS: {ACTIVATION_FUNCTION_OPTIONS}")

    # Cartesian product over all hyper-parameter choices
    for (
        model,
        batch,
        soft_label,
        oversample,
        freeze,
        lr,
        weight,
        hidden,
        dropout,
        activation,
    ) in itertools.product(
        MODEL_OPTIONS,
        BATCH_SIZE_OPTIONS,
        SOFT_LABEL_OPTIONS,
        OVERSAMPLE_OPTIONS,
        FREEZE_ENCODER_OPTIONS,
        LEARNING_RATE_OPTIONS,
        CLASS_WEIGHT_OPTIONS,
        HIDDEN_DIM_OPTIONS,
        DROPOUT_OPTIONS,
        ACTIVATION_FUNCTION_OPTIONS,
    ):

        # Avoid conflicting sampling strategies: cannot oversample AND use class weights
        if oversample and weight is not None:
            continue

        classifier = one_layer_classifier_constructor(
            hidden_dim=hidden,
            dropout_p=dropout,
            activation_func=activation,
        )

        model_opts = create_model_options(model, lr, freeze, classifier, weight, soft_label)
        # Persist key architecture hparams for logging
        model_opts["hidden_dim"] = hidden
        model_opts["activation"] = activation.__class__.__name__ if hasattr(activation, "__class__") else str(activation)

        yield (
            model_opts,
            create_data_options(batch, oversample),
        )


def create_two_hidden_layer_param_iterator():
    """Do all 2-hidden layer options"""
    # Note: This requires HIDDEN_DIM_OPTIONS, DROPOUT_OPTIONS, ACTIVATION_FUNCTION_OPTIONS to be defined
    # These are currently commented out in config.py
    pass 