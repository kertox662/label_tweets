# Refactored Tweet Classification Code

This folder contains a clean, modular refactoring of the original `trainer.py` file. The code has been organized into logical components for better maintainability and reusability.

## File Structure

### Core Components

- **`config.py`** - Configuration constants and parameters

  - Training parameters (epochs, early stopping, etc.)
  - Model options for hyperparameter search
  - Data module parameters

- **`model.py`** - Main model implementation

  - `BertweetClassifier` - PyTorch Lightning module for tweet classification
  - Handles both soft and hard label training
  - Supports custom classifier architectures

- **`data_module.py`** - Data handling

  - `TweetsDataModule` - PyTorch Lightning data module
  - Handles data loading, preprocessing, and splitting
  - Supports oversampling and class weight computation

- **`classifier_constructors.py`** - Classifier architecture helpers

  - `ClassifierConstructor` - Base class for building classifiers
  - `one_layer_classifier_constructor()` - Creates single hidden layer classifiers
  - `two_layer_classifier_constructor()` - Creates double hidden layer classifiers

- **`hyperparameter_search.py`** - Hyperparameter search utilities
  - `param_search()` - Main search function
  - `evaluate_with_params()` - Evaluates a single parameter combination
  - Parameter iterator functions for different search strategies

### Entry Points

- **`main.py`** - Clean entry point for running experiments
- **`__init__.py`** - Package initialization and exports

## Usage

### Basic Usage

```python
from Refactored import BertweetClassifier, TweetsDataModule

# Load data
data_module = TweetsDataModule.read_csv("data/training_data_labelled.csv")
data_module.setup("fit")

# Create model
model = BertweetClassifier(
    transformer_model_name="sentence-transformers/all-mpnet-base-v2",
    learning_rate=1e-4,
    freeze_encoder=True
)

# Train
trainer = pl.Trainer(max_epochs=25)
trainer.fit(model, data_module)
```

### Hyperparameter Search

```python
from Refactored import param_search, create_linear_param_iterator

# Run linear classifier search
param_search(create_linear_param_iterator(), summary_file="hp_search_linear.csv")
```

### Running from Command Line

```bash
# Run the main hyperparameter search
python -m Refactored.main
```

## Key Improvements

1. **Modular Design** - Each component has a single responsibility
2. **Clean Imports** - No duplicate imports or circular dependencies
3. **Removed Duplication** - Eliminated duplicate `BertweetClassifier` definition
4. **Better Organization** - Logical separation of concerns
5. **Maintainable** - Easy to modify individual components
6. **Testable** - Each module can be tested independently

## Configuration

Edit `config.py` to modify:

- Training parameters
- Model options for hyperparameter search
- Data processing parameters

## Adding New Features

- **New Models**: Add to `model.py`
- **New Data Processing**: Extend `data_module.py`
- **New Classifiers**: Add constructors to `classifier_constructors.py`
- **New Search Strategies**: Add iterators to `hyperparameter_search.py`
