import json, glob
from typing import List
from pathlib import Path
from Mixtral import Model, ModelArgs
from sentencepiece import SentencePieceProcessor

import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_unflatten, tree_flatten

class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "▁"

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    @property
    def vocab_size(self) -> int:
        return self._model.get_piece_size()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        out = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + out
        return out

def create_model(folder: str):
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        model_args = ModelArgs(**config)

    model = Model(model_args)

    return model, tokenizer

model, _ = create_model("/Users/gokdenizgulmez/Desktop/mixtral_mlx/v3")
mx.savez("/Users/gokdenizgulmez/Desktop/mixtral_mlx/v3", **dict(tree_flatten(model.parameters())))
