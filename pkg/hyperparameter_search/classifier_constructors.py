import torch.nn as nn


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

        return str({
            "hidden_dim": self.hidden_dim,
            "activation": activ_name,
            "dropout_p": self.dropout_p
        })


def one_layer_classifier_constructor(hidden_dim, dropout_p, activation_func):
    """Returns a function (embedding_dim: int, num_labels: int) -> nn.Module"""
    return ClassifierConstructor(hidden_dim, dropout_p, activation_func)


def two_layer_classifier_constructor(hidden_dim1, hidden_dim2, dropout_p, activation_func):
    """Returns a function (embedding_dim: int, num_labels: int) -> nn.Module"""
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