import pandas as pd

path = "tb_logs"
timestamp = 1762215788

def format(name, additional_columns):
    df = pd.read_csv(f"{path}/classifier_{timestamp}_{name}.csv")
    
    renamed = df.rename(columns = {
        "iteration": "Trial",
        "test_loss": "Testing Loss",
        "test_acc": "Testing Accuracy",
        "test_macro_f1": "F1-Score",
        "test_weighted_f1": "Weighted F1-Score",
        "epochs": "Training Epochs",
        "cm": "Confusion Matrix",
        "train_size": "Training set size",
        "val_size": "Validation set size",
        "test_size": "Test set size",
        **additional_columns
    }).drop("Unnamed: 0", axis=1)

    renamed.to_csv(f"{path}/{name}.csv", index=False)

format("crossval", {"fold": "Fold"})
format("training_agreed", {"label": "Label"})
format("training_disagreed", {"label": "Label"})

# # Cross validation
# crossval = pd.read_csv(f"{path}/classifier_{timestamp}_crossval.csv")
# renamed = crossval.rename(columns = {
#     "iteration": "Trial",
#     "test_loss": "Testing Loss",
#     "test_acc": "Testing Accuracy",
#     "test_macro_f1": "F1-Score",
#     "test_weighted_f1": "Weighted F1-Score",
#     "epochs": "Training Epochs",
#     "cm": "Confusion Matrix",
#     "train_size": "Training set size",
#     "val_size": "Validation set size",
#     "test_size": "Test set size",
#     "fold": "Fold"
# }).drop("Unnamed: 0", axis=1)

# renamed.to_csv(f"{path}/crossval.csv", index=False)

# # Agreement
# agreed = pd.read_csv(f"{path}/classifier_{timestamp}_training_agreed.csv")
# renamed = agreed.rename(columns = {
#     "iteration": "Trial",
#     "test_loss": "Testing Loss",
#     "test_acc": "Testing Accuracy",
#     "test_macro_f1": "F1-Score",
#     "test_weighted_f1": "Weighted F1-Score",
#     "epochs": "Training Epochs",
#     "cm": "Confusion Matrix",
#     "train_size": "Training set size",
#     "val_size": "Validation set size",
#     "test_size": "Test set size",
#     "label": "Label"
# }).drop(["Unnamed: 0"], axis=1)

# renamed.to_csv(f"{path}/agreed.csv", index=False)
