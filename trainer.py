import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import itertools
import pandas as pd
import re
from ftfy import fix_text
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler



class BertweetClassifier(pl.LightningModule):
    def __init__(
        self,
        transformer_model_name: str = "vinai/bertweet-base",
        num_labels: int = 3,
        learning_rate: float = 5e-5,
        warmup_ratio: float = 0.10,
        freeze_encoder: bool = True,
        use_soft_labels: bool = False,
        classifier_constructor = None, # function (embedding_dim: int, num_labels: int) -> nn.Module
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.use_soft_labels = use_soft_labels
        # 1) encoder
        self.encoder = SentenceTransformer(transformer_model_name)

        # 2) Set the classifier that will run on the embedding
        #    If it is provided, just use that, otherwise make
        #    a simple linear layer.
        hidden = self.encoder.get_sentence_embedding_dimension()
        if classifier_constructor is not None:
            self.classifier = classifier_constructor(hidden, num_labels)
        else:
            self.classifier = nn.Linear(hidden, num_labels)

        # 3) (optionally) freeze encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        cls_weights = None if class_weights is None else class_weights.to(self.device)
        if self.use_soft_labels:
            self.loss_fn = nn.BCEWithLogitsLoss(weight=cls_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=cls_weights)

    # Forward expects ONLY the list[str] texts
    def forward(self, texts: list[str]):
        embeds = self.encoder.encode(texts, convert_to_tensor=True, device=self.device)
        return self.classifier(embeds)

    # ------------------------------------------------------------------
    # Shared step to avoid duplication
    def _shared_step(self, batch, stage: str):
        texts, labels, soft_labels, true_AR, true_MB = batch
        logits = self(texts)
        if self.use_soft_labels:
            loss = self.loss_fn(logits, soft_labels.to(self.device))
        else:
            loss = self.loss_fn(logits, labels.to(self.device))

        # Get probabilities
        preds = logits.argmax(1)
        acc_ar = (preds == true_AR.to(self.device)).float().mean()
        acc_mb = (preds == true_MB.to(self.device)).float().mean()
        # true_AR, true_MB, preds = true_AR.cpu().numpy(), true_MB.cpu().numpy(), preds.cpu().numpy()

        f1_ar = f1_score(true_AR.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        f1_mb = f1_score(true_MB.cpu().numpy(), preds.cpu().numpy(), average='weighted')

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage=="train"), on_epoch=True)
        self.log(f"{stage}_acc_ar", acc_ar, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_f1_ar", f1_ar, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc_mb", acc_mb, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_f1_mb", f1_mb, prog_bar=True, on_epoch=True, on_step=False)

        # Optionally: return predictions for logging or downstream use
        # Return as a dict for Lightning
        return {
            "loss": loss,
            "preds": preds,
            "true_AR": true_AR,
            "true_MB": true_MB,
            "acc_ar": acc_ar,
            "acc_mb": acc_mb,
            "f1_ar": f1_ar,
            "f1_mb": f1_mb,
        }
    # Lightning API -----------------------------------------------------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    # def test_step(self, batch, batch_idx):
    #     # Re-use the same bookkeeping you used for val:
    #     return self._shared_step(batch, stage="test")
    def on_test_epoch_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        result = self._shared_step(batch, stage="test")
        # Append for use in on_test_epoch_end
        self.test_outputs.append({
            "preds": result["preds"].cpu(),
            "true_AR": result["true_AR"].cpu(),
            "true_MB": result["true_MB"].cpu(),
            "acc_ar": result["acc_ar"],
            "acc_mb": result["acc_mb"],
            "f1_ar": result["f1_ar"],
            "f1_mb": result["f1_mb"],
        })
        return result

    def on_test_epoch_end(self):
        preds    = torch.cat([x["preds"]    for x in self.test_outputs])
        true_AR  = torch.cat([x["true_AR"]  for x in self.test_outputs])
        true_MB  = torch.cat([x["true_MB"]  for x in self.test_outputs])

        acc_ar = (preds == true_AR).float().mean().item()
        acc_mb = (preds == true_MB).float().mean().item()
        from sklearn.metrics import f1_score
        f1_ar  = f1_score(true_AR.numpy(), preds.numpy(), average='weighted')
        f1_mb  = f1_score(true_MB.numpy(), preds.numpy(), average='weighted')

        self.log("test_acc_ar", acc_ar)
        self.log("test_acc_mb", acc_mb)
        self.log("test_f1_ar", f1_ar)
        self.log("test_f1_mb", f1_mb)
        # Save results for outside access if needed
        self.test_metrics = {
            "test_acc_ar": acc_ar,
            "test_acc_mb": acc_mb,
            "test_f1_ar": f1_ar,
            "test_f1_mb": f1_mb,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # `batch` is just a list[str] from your predict_dataloader
        texts = batch
        logits = self(texts)
        probs  = torch.softmax(logits, dim=-1)
        return probs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        total_steps  = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return {
            "optimizer":  optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",      # step every batch
                "frequency": 1,
                "name": "linear-warmup",
            },
        }
    
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from transformers import get_linear_schedule_with_warmup

class BertweetClassifier(pl.LightningModule):
    def __init__(
        self,
        transformer_model_name: str = "vinai/bertweet-base",
        num_labels: int = 3,
        learning_rate: float = 5e-5,
        warmup_ratio: float = 0.10,
        freeze_encoder: bool = True,
        use_soft_labels: bool = False,
        classifier_constructor = None, # function (embedding_dim: int, num_labels: int) -> nn.Module
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.use_soft_labels = use_soft_labels
        # 1) encoder
        self.encoder = SentenceTransformer(transformer_model_name)

        # 2) Set the classifier that will run on the embedding
        #    If it is provided, just use that, otherwise make
        #    a simple linear layer.
        hidden = self.encoder.get_sentence_embedding_dimension()
        if classifier_constructor is not None:
            self.classifier = classifier_constructor(hidden, num_labels)
        else:
            self.classifier = nn.Linear(hidden, num_labels)

        # 3) (optionally) freeze encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        cls_weights = None if class_weights is None else class_weights.to(self.device)
        if self.use_soft_labels:
            self.loss_fn = nn.BCEWithLogitsLoss(weight=cls_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=cls_weights)

    # Forward expects ONLY the list[str] texts
    def forward(self, texts: list[str]):
        embeds = self.encoder.encode(texts, convert_to_tensor=True, device=self.device)
        return self.classifier(embeds)

    # ------------------------------------------------------------------
    # Shared step to avoid duplication
    def _shared_step(self, batch, stage: str):
        texts, labels, soft_labels, true_AR, true_MB = batch
        logits = self(texts)
        if self.use_soft_labels:
            loss = self.loss_fn(logits, soft_labels.to(self.device))
        else:
            loss = self.loss_fn(logits, labels.to(self.device))

        # Get probabilities
        preds = logits.argmax(1)
        acc_ar = (preds == true_AR.to(self.device)).float().mean()
        acc_mb = (preds == true_MB.to(self.device)).float().mean()
        # true_AR, true_MB, preds = true_AR.cpu().numpy(), true_MB.cpu().numpy(), preds.cpu().numpy()

        f1_ar = f1_score(true_AR.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        f1_mb = f1_score(true_MB.cpu().numpy(), preds.cpu().numpy(), average='weighted')

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage=="train"), on_epoch=True)
        self.log(f"{stage}_acc_ar", acc_ar, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_f1_ar", f1_ar, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc_mb", acc_mb, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_f1_mb", f1_mb, prog_bar=True, on_epoch=True, on_step=False)

        # Optionally: return predictions for logging or downstream use
        # Return as a dict for Lightning
        return {
            "loss": loss,
            "preds": preds,
            "true_AR": true_AR,
            "true_MB": true_MB,
            "acc_ar": acc_ar,
            "acc_mb": acc_mb,
            "f1_ar": f1_ar,
            "f1_mb": f1_mb,
        }
    # Lightning API -----------------------------------------------------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    # def test_step(self, batch, batch_idx):
    #     # Re-use the same bookkeeping you used for val:
    #     return self._shared_step(batch, stage="test")
    def on_test_epoch_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        result = self._shared_step(batch, stage="test")
        # Append for use in on_test_epoch_end
        self.test_outputs.append({
            "preds": result["preds"].cpu(),
            "true_AR": result["true_AR"].cpu(),
            "true_MB": result["true_MB"].cpu(),
            "acc_ar": result["acc_ar"],
            "acc_mb": result["acc_mb"],
            "f1_ar": result["f1_ar"],
            "f1_mb": result["f1_mb"],
        })
        return result

    def on_test_epoch_end(self):
        preds    = torch.cat([x["preds"]    for x in self.test_outputs])
        true_AR  = torch.cat([x["true_AR"]  for x in self.test_outputs])
        true_MB  = torch.cat([x["true_MB"]  for x in self.test_outputs])

        acc_ar = (preds == true_AR).float().mean().item()
        acc_mb = (preds == true_MB).float().mean().item()
        from sklearn.metrics import f1_score
        f1_ar  = f1_score(true_AR.numpy(), preds.numpy(), average='weighted')
        f1_mb  = f1_score(true_MB.numpy(), preds.numpy(), average='weighted')

        self.log("test_acc_ar", acc_ar)
        self.log("test_acc_mb", acc_mb)
        self.log("test_f1_ar", f1_ar)
        self.log("test_f1_mb", f1_mb)
        # Save results for outside access if needed
        self.test_metrics = {
            "test_acc_ar": acc_ar,
            "test_acc_mb": acc_mb,
            "test_f1_ar": f1_ar,
            "test_f1_mb": f1_mb,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # `batch` is just a list[str] from your predict_dataloader
        texts = batch
        logits = self(texts)
        probs  = torch.softmax(logits, dim=-1)
        return probs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        total_steps  = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return {
            "optimizer":  optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",      # step every batch
                "frequency": 1,
                "name": "linear-warmup",
            },
        }
    
SUMMARY_FILE = "search_summary.csv"
PARAM_SEARCH_LOG_NAME = "hp_search"
DEFAULT_TRIALS_PER_PARAMS = 1
EPOCHS = 25
EARLY_STOPPING_EPOCH = 5
EARLY_STOPPING_MIN_DELTA = .01
datamodule_params = {
     "batch_size": 64,
     "target_col": 'AR',
     "test_size":0.2,
     "validation_size":0.2,
     "oversample":True,
     "random_state":2025
}

base_datamodule_params = {
     "target_col": 'AR',
     "test_size":0.2,
     "validation_size":0.2,
     "random_state":2025
}
MODEL_OPTIONS = [
    # "digio/Twitter4SSE",
    "sentence-transformers/all-mpnet-base-v2",
    # "peulsilva/sentence-transformer-trained-tweet",
    # "vinai/bertweet-base",
]
SOFT_LABEL_OPTIONS = [True, False]
# HIDDEN_DIM_OPTIONS = [64]
# DROPOUT_OPTIONS = [0.5]
# ACTIVATION_FUNCTION_OPTIONS = [nn.ReLU()]
# ACTIVATION_FUNCTION_OPTIONS = [
#     nn.ReLU(), nn.Tanh(), nn.Hardswish(),
#     nn.LeakyReLU(),
# ]
LEARNING_RATE_OPTIONS = [1e-5, 1e-4, 1e-3]
FREEZE_ENCODER_OPTIONS = [True]
BATCH_SIZE_OPTIONS = [64]
OVERSAMPLE_OPTIONS = [False]
CLASS_WEIGHT_OPTIONS = [True, False]

# OUTPUT_COLUMNS = ["model_params", "data_params", "test_acc_AR", "test_acc_MB", "test_loss"]

class ClassifierConstructor:
    def __init__(self, hidden_dim, dropout_p, activation):
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.activation = activation

    def __call__(self, embedding_dim, num_labels):
        return nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_dim),
            self.activation,
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(self.dropout_p),

            nn.Linear(self.hidden_dim, num_labels)
        )

    def __repr__(self):
        activ_name = ""
        if type(self.activation) is nn.ReLU:
            activ_name = "ReLU"
        elif type(self.activation) is nn.LeakyReLU:
            activ_name = "LeakyReLU"

        return str(
            {
                "hidden_dim": self.hidden_dim,
                "activation": activ_name,
                "dropout_p": self.dropout_p
            }
        )

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

def param_search(param_iterator, trials_per_param = DEFAULT_TRIALS_PER_PARAMS, summary_file = "hp_search_summary.csv"):
    # results_df = pd.DataFrame([])
# , columns=OUTPUT_COLUMNS
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
                # "model_params": str(model_params),
                # "data_params": str(data_params),
                **model_params,
                **results
            }
            result.append(trial_results)
        results_df = pd.DataFrame(result)
        results_df.to_csv(summary_file)

# Will return a function (embedding_dim: int, num_labels: int) -> nn.Module
def one_layer_classifier_constructor(hidden_dim, dropout_p, activation_func):
    return ClassifierConstructor(hidden_dim, dropout_p, activation_func)

# Will return a function (embedding_dim: int, num_labels: int) -> nn.Module
def two_layer_classifier_constructor(hidden_dim1, hidden_dim2, dropout_p, activation_func):
    def constructor(embedding_dim, num_labels):
        return nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim1),
            activation_func,
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dim1, hidden_dim2),
            activation_func,
            nn.BatchNorm1d(hidden_dim2),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dim2, num_labels)
        )

    return constructor

def create_model_options(model_name, learning_rate, freeze_encoder, classifier_constructor, class_weights, soft_label):
    return {
        "transformer_model_name": model_name,
        "learning_rate": learning_rate,
        "freeze_encoder": freeze_encoder,
        "classifier_constructor": classifier_constructor,
        "class_weights": class_weights,
        "use_soft_labels":soft_label,
    }

def create_data_options(batch_size, oversample):
    return {
        "batch_size": batch_size,
        "oversample": oversample,
        **base_datamodule_params,
    }


def create_linear_param_iterator():
    # Do all linear layer options
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
    # Do all 1-hidden layer options
    print(f"MODEL_OPTIONS: {MODEL_OPTIONS}")
    print(f"HIDDEN_DIM_OPTIONS: {HIDDEN_DIM_OPTIONS}")
    print(f"DROPOUT_OPTIONS: {DROPOUT_OPTIONS}")
    print(f"ACTIVATION_FUNCTION_OPTIONS: {ACTIVATION_FUNCTION_OPTIONS}")
    print(f"BATCH_SIZE_OPTIONS: {BATCH_SIZE_OPTIONS}")
    print(f"OVERSAMPLE_OPTIONS: {OVERSAMPLE_OPTIONS}")
    print(f"LEARNING_RATE_OPTIONS: {LEARNING_RATE_OPTIONS}")
    print(f"CLASS_WEIGHT_OPTIONS: {CLASS_WEIGHT_OPTIONS}")
    for (model, hidden, dropout, activ, batch, oversample, freeze, lr, weight) in itertools.product(
        MODEL_OPTIONS, HIDDEN_DIM_OPTIONS, DROPOUT_OPTIONS, ACTIVATION_FUNCTION_OPTIONS,
        BATCH_SIZE_OPTIONS, OVERSAMPLE_OPTIONS, FREEZE_ENCODER_OPTIONS, LEARNING_RATE_OPTIONS, CLASS_WEIGHT_OPTIONS,
    ):
        print("----------------------------------------------------------------------------------------------------\n")
        print(f"hidden: {hidden}, dropout: {dropout}, activ: {activ}, batch: {batch}, oversample: {oversample}")
        print(f"freeze: {freeze}, lr: {lr}, weight: {weight}")
        if oversample and weight is not None:
            continue

        classifier = one_layer_classifier_constructor(hidden, dropout, activ)

        yield create_model_options(model, lr, freeze, classifier, weight), create_data_options(batch, oversample)

def create_two_hidden_layer_param_iterator():
    # Do all 2-hidden layer options
    for (model, hidden1, hidden2, dropout, activ, batch, oversample, freeze, lr, weight) in itertools.product(
        MODEL_OPTIONS, HIDDEN_DIM_OPTIONS, HIDDEN_DIM_OPTIONS, DROPOUT_OPTIONS, ACTIVATION_FUNCTION_OPTIONS,
        BATCH_SIZE_OPTIONS, OVERSAMPLE_OPTIONS, FREEZE_ENCODER_OPTIONS, LEARNING_RATE_OPTIONS, CLASS_WEIGHT_OPTIONS,
    ):
        if oversample and weight is not None:
            continue

        classifier = two_layer_classifier_constructor(hidden1, hidden2, dropout, activ)

        yield create_model_options(model, lr, freeze, classifier, weight), create_data_options(batch, oversample)

torch.set_float32_matmul_precision('high')
param_search(create_linear_param_iterator(), summary_file="hp_search_linear.csv")
# param_search(create_one_hidden_layer_param_iterator(), summary_file="hp_search_1hidden.csv")
# param_search(create_two_hidden_layer_param_iterator(), summary_file="hp_search_2hidden.csv")

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler

from preprocessing_steps.data_cleanup import *

class TweetsDataModule(pl.LightningDataModule):
    def __init__(self, data: pd.DataFrame, batch_size: int = None, target_col: str = 'AR',
                 test_size=0.2, validation_size=0.2, oversample=False, random_state=2025):
        super().__init__()
        self.data = data.copy()
        self.batch_size = batch_size
        self.target_col = target_col

        self.label2id: dict[str, int] = {}
        self.id2label: list[str]      = []

        self._test_size = test_size
        self._validation_size = validation_size
        self._oversample = oversample
        self._random_state = random_state
    
    @classmethod
    def read_csv(self, filename: str = "data/training_data_labelled.csv", remove_disagreements: bool = False,
                 batch_size: int = None, target_col: str = 'AR', test_size=0.2, validation_size=0.2,
                 oversample=False, random_state=2025):
        tweets_labeled = pd.read_csv(filename)
        tweets_labeled['AR'].replace({4:2}, inplace=True)
        tweets_labeled['MB'].replace({4:2}, inplace=True)

        if remove_disagreements:
            tweets_labeled = tweets_labeled[tweets_labeled["AR"] == tweets_labeled["MB"]]

        return TweetsDataModule(
            tweets_labeled, batch_size, target_col, test_size, validation_size,
            oversample, random_state
        )

    def setup(self, stage: str=None):
        # Assign train/val datasets for use in dataloaders
        if stage != "fit":
            return
        self.data = preprocess_text(self.data, text_col="text")
        uniques          = sorted(self.data[self.target_col].unique())
        self.num_labels = len(uniques)
        self.label2id    = {lbl: i for i, lbl in enumerate(uniques)}
        self.id2label    = uniques                            # same order
        self.data["y"]   = self.data[self.target_col].map(self.label2id)    
        self.data['AR_id'] = self.data['AR'].map(self.label2id)
        self.data['MB_id'] = self.data['MB'].map(self.label2id)
        self.data['soft_label'] = self.data.apply(self.make_soft_label, axis=1)

        self._train_test_split()
        
    def _train_test_split(self):
        train, self.test = train_test_split(self.data, train_size=1-self._test_size, random_state=self._random_state, shuffle=True)
        self.train, self.val = train_test_split(train, train_size=1-self._validation_size, random_state=self._random_state, shuffle=True)
        if self._oversample:
            ros = RandomOverSampler(random_state=self._random_state)
            # Extract features and label
            X_res, y_res = ros.fit_resample(self.train, self.train["y"])
            self.train = X_res.copy()
            self.train["y"] = y_res
        
        print(f"Shape of training/validation/test: {self.train.shape[0]}/{self.val.shape[0]}/{self.test.shape[0]}")
        print(f"Label distribution for training data: {self.train.y.value_counts()}")


    def make_soft_label(self, row):
        arr = np.zeros(self.num_labels)
        arr[self.label2id[row['AR']]] += 1
        arr[self.label2id[row['MB']]] += 1
        arr /= arr.sum()  # divide by 2
        return arr

    def _zip(self, data):
        return list(zip(
            data["clean_text"], 
            data["y"],
            data["soft_label"],
            data["AR_id"], 
            data["MB_id"]
        ))
    
    def _dataloader(self, dataset, shuffle, drop_last):
        data = self._zip(dataset)
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: (
                [t for t, _, _, _, _ in batch],
                torch.tensor([y for _, y, _, _, _ in batch], dtype=torch.long),
                torch.tensor([l for _, _, l, _, _ in batch], dtype=torch.float32),
                torch.tensor([ar for _, _, _, ar, _ in batch], dtype=torch.long),
                torch.tensor([mb for _, _, _, _, mb in batch], dtype=torch.long),
            ),
            drop_last=drop_last,
        )

    def train_dataloader(self):
        return self._dataloader(dataset=self.train, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._dataloader(dataset=self.val, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._dataloader(dataset=self.test, shuffle=False, drop_last=False)
    
    def predict_dataloader(self):
        """
        Returns a DataLoader for inference.
        - If you called `setup("predict")`, it will use `self.test`.
        - If you created `self.predict` manually (e.g., a new DataFrame without labels),
          that takes precedence.
        Collate fn yields a *list[str]* so the model’s forward just receives raw texts.
        """
        dataset = getattr(self, "predict", None)
        if dataset is None:          # fall back to the held-out test set
            dataset = self.test

        texts = list(dataset["clean_text"])

        return DataLoader(
            texts,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch  # returns List[str] per batch
        )
    
    def train_class_weights(self):
        labels = np.array(self.train["y"])
        weights = compute_class_weight('balanced', classes=np.arange(3), y=labels)
        return torch.Tensor(weights)