import json
from pathlib import Path
from dataclasses import dataclass

import mlx.optimizers as optim
from train_that_mf import train
from create_model import create_model

model_load_path = "/Users/gokdenizgulmez/Desktop/mixtral_mlx/v3"
pretrained_model_save_path = "/Users/gokdenizgulmez/Desktop/mixtral_mlx/v3/pretrained_model.npz"

@dataclass
class PreTrainArgs():
    data: str = "/Users/gokdenizgulmez/Desktop/mixtral_mlx/pretrain_data" # Path of the Dataset folder
    lora_layers: int = 0 # Number of layers to fine-tune. Set to "0" if you want to train the full model. Default is 16

    learning_rate: float = 1e-5 # Adam's learning rate
    weight_decay: float = 1e-5

    train: bool = True # Train the model
    test: bool = True # Test the model

    batch_size: int = 1 # Minibatch size, min 2.
    val_batches: int = 1 # Number of validation batches, -1 uses the entire validation set, min 2.
    test_batches: int = 1 # Number of test set batches, -1 uses the entire test set, min 2.

    iters: int = 10 # Iterations to train for

    steps_per_report: int = 1 # Number of training steps between loss reporting
    steps_per_eval: int = 10 # Number of training steps between validations

    # save_every: int = 10 # Save the model every N iterations

    resume_adapter_file = None # Load path to resume training with the given adapter weights
    # adapter_file: str = "pretrained_model.npz" # Save/load path for the trained adapter weights

    max_position_embeddings: int = 32

# Create the model and load the tokenizer
model, tokenizer = create_model(model_load_path)
# print(model)
print("Created Model and Loaded Tokenizer")

class Dataset:
    """
    Light-weight wrapper to hold lines from a jsonl file
    """

    def __init__(self, path: Path, key: str = "text"):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        return len(self._data)


def load(args):
    def load_and_check(name):
        dataset_path = Path(args.data) / f"{name}.jsonl"
        try:
            return Dataset(dataset_path)
        except Exception as e:
            print(f"Unable to build dataset Boiiii {dataset_path} ({e})")
            raise

    names = ("train", "val", "test")
    train, valid, test = (load_and_check(n) for n in names)

    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test

print("Loading datasets")
train_set, valid_set, test_set = load(PreTrainArgs)
print("Datasets loaded")

# Creqate the Optimizer
opt = optim.AdamW(learning_rate=PreTrainArgs.learning_rate, weight_decay=PreTrainArgs.weight_decay)

# Train Model
train(model=model, train_set=train_set, val_set=valid_set, optimizer=opt, tokenizer=tokenizer, args=PreTrainArgs)
