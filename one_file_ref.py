# Copyright © 2023 Apple Inc. & Gökdeniz Gülmez

import os
import math
import json
import time
import glob
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
from simple_parsing.helpers import Serializable
from typing import List, Optional, Tuple, Generator, Dict, Union

import numpy as np

from lora import LoRALinear

import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from sentencepiece import SentencePieceProcessor



########################################################################################################################
###################### Tokenizer Part only used in tloading the model weights ##########################################
########################################################################################################################

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

# Create the Tokenizer
tokenizer = Tokenizer("/Users/gokdenizgulmez/Desktop/mixtral_mlx/mixtral/tokenizer.model")

print(f"Loaded Tokenzer with eos_id {tokenizer.eos_id} and a vocab size of {tokenizer.vocab_size}")



########################################################################################################################
###################### Model Args / Params #############################################################################
########################################################################################################################

@dataclass
class MoeArgs(Serializable):
    num_experts_per_tok: int = 2
    num_local_experts: int = 8

@dataclass
class ModelArgs(Serializable):
    architecture: str = "MixtralForCausalLM"
    model_type: str = "mixtral"
    creators = ["Gökdeniz Gülmez", " Apple Inc and it's research team"]
    model_save_path: str = "/Users/gokdenizgulmez/Desktop/mixtral_mlx/mixtral/weights.npz"

    hidden_act: str = "silu"

    hidden_size: int = 1024 # dim
    intermediate_size: int = 4096 # hidden dim
    max_position_embeddings: int = 32 # Max context size

    num_hidden_layers: int = 12 # Transfomrer Layers

    num_attention_heads: int = 16
    num_key_value_heads: int = 8

    # head_dim: int = hidden_size // num_attention_heads = 1024 // 32

    rms_norm_eps: float = 1e-06

    vocab_size: int = tokenizer.vocab_size

    rope_traditional: bool = True
    rope_theta: float = 1e6 # RoPE Base
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    num_experts_per_tok: int = 2
    num_local_experts: int = 8

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

@dataclass
class PreTrainArgs():
    data: str = "/Users/gokdenizgulmez/Desktop/mixtral_mlx/pretrain_data" # Path of the Dataset folder
    lora_layers: int = 0 # Number of layers to fine-tune. Set to "0" if you want to train the full model. Default is 16

    learning_rate: float = 1e-5 # Adam's learning rate
    weight_decay: float = 1e-5

    train: bool = True # Train the model
    test: bool = True # Test the model

    batch_size: int = 2 # Minibatch size, min 2.
    val_batches: int = 2 # Number of validation batches, -1 uses the entire validation set, min 2.
    test_batches: int = 2 # Number of test set batches, -1 uses the entire test set, min 2.

    iters: int = 1000 # Iterations to train for

    steps_per_report: int = 10 # Number of training steps between loss reporting
    steps_per_eval: int = 10 # Number of training steps between validations

    save_every: int = 10 # Save the model every N iterations

    resume_adapter_file = None # Load path to resume training with the given adapter weights
    adapter_file: str = "pretrained_model.npz" # Save/load path for the trained adapter weights

@dataclass
class FineTuneArgs():
    data: str = "/Users/gokdenizgulmez/Desktop/mixtral_mlx/finetune_data"
    lora_layers: int = 16

    learning_rate: float = 1e-5
    weight_decay: float = 1e-5

    train: bool = True
    test: bool =True

    test_batches: int = 2
    batch_size: int = 4
    val_batches: int = 2

    iters: int = 200

    steps_per_report: int = 2
    steps_per_eval: int = 100

    save_every: int = 100

    resume_adapter_file = None
    adapter_file: str = "/Users/gokdenizgulmez/Desktop/mixtral_mlx/mixtral/finetuned_adapter.npz"



########################################################################################################################
###################### RMS Normalisierung ##############################################################################
########################################################################################################################

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output



########################################################################################################################
###################### Rotairy Positional Embedding ####################################################################
########################################################################################################################

class RoPE(nn.RoPE):
    def __init__(self, dims: int, traditional: bool = False):
        super().__init__(dims, traditional)

    def __call__(self, x, offset: int = 0):
        shape = x.shape
        x = mx.reshape(x, (-1, shape[-2], shape[-1]))
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N, self.dims, offset=offset, base=1000000, dtype=x.dtype
        )

        rope = (
            self._compute_traditional_rope if self.traditional else self._compute_rope
        )
        rx = rope(costheta, sintheta, x)

        return mx.reshape(rx, shape)



########################################################################################################################
###################### Multi Head Attention ############################################################################
########################################################################################################################

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.max_position_embeddings = args.max_position_embeddings
        self.rope_theta = args.rope_theta

        self.repeats = self.num_heads // self.num_key_value_heads

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # self.rope = RoPE(args.head_dim, traditional=args.rope_traditional) # this uses the defined RoPE class abouve
        self.rope = nn.RoPE(self.head_dim, traditional=args.rope_traditional, base=args.rope_theta) # For using the build in RoPE embedding from mlx

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Tuple[mx.array, mx.array]] = None) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        # For Broadcast
        if self.repeats > 1:
            keys = mx.repeat(keys, self.repeats, axis=1)
            values = mx.repeat(values, self.repeats, axis=1)

        # Cache the lasty generated Logits for effitience
        if cache is not None:
            key_cache, value_cache = cache

            queries = self.rope(queries, offset=key_cache.shape[2])

            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)

            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)

        # Set Casual Mask if set
        if mask is not None:
            scores += mask

        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output), (keys, values)



########################################################################################################################
###################### Mixtral Sparse Top2 MLP FeedForward Module ######################################################
########################################################################################################################

class MoeFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ffn_dim = args.intermediate_size
        self.hidden_dim = args.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = nn.silu # Silu Activation Function

    def __call__(self, x: mx.array) -> mx.array:

        current_hidden_states = self.act_fn(self.w1(x)) * self.w3(x)
        current_hidden_states = self.w2(current_hidden_states)

        return current_hidden_states

    # def __call__(self, x) -> mx.array:
    #     return self.w2(nn.silu(self.w1(x)) * self.w3(x)) # For a one liner



########################################################################################################################
###################### MoE Sparse FeedForward Module Block where the Experts are defined ###############################
########################################################################################################################

class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.ffn_dim = args.intermediate_size

        self.num_experts = args.num_local_experts
        self.num_experts_per_tok = args.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = [MoeFeedForward(args=args) for _ in range(self.num_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape # [batch_size, max_context_length?, dim]
        x = x.reshape(-1, x.shape[-1])

        gates = self.gate(x)

        # Top_K selection. Selects the indices of the top num_experts_per_tok experts for each token.
        # This is achieved by first negating the gates scores to use argpartition for descending order selection, then partitioning and slicing to keep only the top num_experts_per_tok indices.
        # Since num_experts_per_tok is set to select the top 2 experts, this step effectively picks the indices of the top 2 experts for each token.
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne])  # TODO remove it once we figure out how to fine tune TopK in MOE

        # Spftmax Scaling
        # The scores for the selected experts are then scaled by applying a softmax operation.
        # This operation is performed on the scores obtained from gates that correspond to the selected top num_experts_per_tok experts' indices.
        scores = mx.softmax(
            mx.take_along_axis(gates, inds, axis=-1).astype(mx.float32),
            axis=-1,
        ).astype(gates.dtype)

        # Checks if model is in Training
        if self.training: # During training, this block performs the actual routing of tokens to the selected experts.
            mx.eval(inds) # It first evaluates the indices
            inds = np.array(inds) # Then converts them into np array
            y = mx.zeros((x.shape[0], ne, x.shape[-1])) # A zero tensor 'y' is created with dimensions [batch_size * max_context_length, num_experts_per_tok, hidden_dim].

            # For each expert, it checks if there are any tokens assigned to it by looking at the indices.
            for e, expert in enumerate(self.experts):
                idx1, idx2 = map(mx.array, np.where(inds == e))

                # If tokens are assigned, it processes those tokens through the expert and updates the corresponding positions in 'y'.
                if idx1.size == 0:
                    continue

                # After processing all experts, it scales the outputs by the softmax scores and sums across the experts' dimension to combine their contributions.
                y[idx1, idx2] = expert(x[idx1])

            y = (y * scores[:, :, None]).sum(axis=1)  # This summed output is then reshaped back to the original input shape.
            # This mechanism allows the model to utilize different subsets of experts for different parts of the input, potentially increasing model capacity and flexibility.
        # If not in Training then jsut set the model to evaluation aka for generation
        else:
            # Expert Calls and Weighted Sum
            # For each token, the selected top experts are called with the input xt, and their outputs are combined according to the softmax-scaled scores.
            # The combination is a weighted sum where each expert's output is multiplied by its corresponding score before summing them together.
            y = [] # initializes an empty list to store the processed outputs for each input token.

            for xt, st, it in zip(x, scores, inds.tolist()): # Loop Over Tokens and Corresponding Scores and Indices
                yt = mx.concatenate([self.experts[e](xt)[:, None] for e in it], axis=-1)
                yt = (yt * st).sum(axis=-1) # performs an element-wise multiplication of the concatenated expert outputs

                y.append(yt[None, :]) # Appending Processed Output, takes the summed output for the current token, adds a new axis at the beginning ([None, :])
            y = mx.concatenate(y) # Final output concatination
            # Concatenates all the processed token outputs along the first axis, transforming the list of token outputs into a single output tensor y.
            # This tensor is structured to match the original batch of inputs in terms of sequence length but is now enriched by the selective, weighted contributions of the top experts for each token.

        return y.reshape(orig_shape)



########################################################################################################################
###################### MoE Transformre Block/Layer #####################################################################
########################################################################################################################

class MixtralDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size

        self.self_attn = Attention(args) # Multi Head Attention Layer

        self.block_sparse_moe = MixtralSparseMoeBlock(args) # The FeedForward
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps) # FeedForward RMS Normalisierung
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Tuple[mx.array, mx.array]] = None) -> mx.array:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.block_sparse_moe(self.post_attention_layernorm(h))

        out = h + r

        return out, cache

# Mixtral Transformer Model
class MixtralModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = [MixtralDecoderLayer(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        mask = None
        T = h.shape[1]

        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.norm(h), cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.creators = args.creators
        self.model = MixtralModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out, cache = self.model(inputs, cache)
        return self.lm_head(out), cache

    @property
    def layers(self):
        return self.model.layers



########################################################################################################################
###################### Loads the Model weights and the Tokenzer and its config file ####################################
########################################################################################################################

def load_model(folder: str):
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)
        model_args = ModelArgs(**config)
    weight_files = glob.glob(str(model_path / "weights.*.npz"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())
    weights = tree_unflatten(list(weights.items()))
    model = Model(model_args)
    if quantization is not None:
        # TODO: Quantize gate matrices when < 32 tiles supported
        quantization["linear_class_predicate"] = (
            lambda m: isinstance(m, nn.Linear) and m.weight.shape[0] != 8
        )
        nn.QuantizedLinear.quantize_module(model, **quantization)

    model.update(weights)
    return model, tokenizer



########################################################################################################################
###################### Generates based on a tokenized input text and a given temperature ###############################
########################################################################################################################

def generate(prompt: mx.array, model: nn.Module, temp: float = 0.0) -> Generator[mx.array, None, None]:
    """
    Generate text based on the given prompt and model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling. If temp is 0, use max sampling.

    Yields:
        mx.array: The generated text.
    """

    # Defines a local function 'sample' that takes logits (the raw, unnormalized predictions from a model) as input.
    def sample(logits):
        if temp == 0: # If the temperature is 0, perform argmax sampling, which selects the most likely next token.
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp)) # If the temperature is not 0, adjust the logits according to the temperature and sample from the distribution.

    # Forward Pass though the model
    logits, cache = model(prompt[None]) # Runs the model once with the initial prompt to get the first set of logits and a cache (if applicable).
    y = sample(logits[:, -1, :]) # Samples the next token based on the last set of logits.
    yield y # Yields the first generated token.

    while True: # Enters a loop to generate subsequent tokens.
        # For each new token, the model is run with the previously generated token and the updated cache.
        # Forward Pass though the model
        logits, cache = model(y[:, None], cache) # Squeezes the logits to remove the singleton dimension and samples the next token based on these logits.
        y = sample(logits.squeeze(1))
        yield y # Yields the next generated token.


def generate_raw(prompt: mx.array, model):
    def sample(logits):
        return mx.argmax(logits, axis=-1)

    logits, cache = model(prompt[None]) # Forward Pass
    y = sample(logits[:, -1, :])
    yield y

    while True:
        logits, cache = model(y[:, None], cache)
        y = sample(logits.squeeze(1))
        yield y

def generate_one_token(prompt: mx.array, model: nn.Module, temp: float = 0.0) -> mx.array:
    """
    Generate a single text token based on the given prompt and model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling. If temp is 0, use max sampling.

    Returns:
        mx.array: The generated single token.
    """
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp), axis=-1)

    # Run the model once with the initial prompt to get the first set of logits.
    logits, _ = model(prompt[None])
    # Sample the next token based on the last set of logits.
    y = sample(logits[:, -1, :])
    # Return the first generated token instead of yielding.
    return y

def just_generate_one_god_damn_token(prompt: mx.array, model: nn.Module) -> mx.array:
    """
    Generate a single text token based on the given prompt and model, using the most likely prediction.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.

    Returns:
        mx.array: The generated single token.
    """
    print(f"Tokenized prompt: {prompt}")
    # Run the model once with the initial prompt to get the first set of logits.
    logits, _ = model(prompt[None])
    print(f"Generated logits: {logits}")
    # Use argmax to select the most likely next token.
    y = mx.argmax(logits[:, -1, :], axis=-1)
    print(f"Selected most likely Token: {y}")
    # Return the first generated token.
    return y

def just_generate_one_god_damn_token_but_full_to_test(prompt: str, model: nn.Module, tokenizer: Tokenizer) -> mx.array:
    """
    Generate a single text token based on the given prompt and model, using the most likely prediction.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.

    Returns:
        mx.array: The generated single token.
    """

    print(f"Raw prompt: {prompt}")

    token = mx.array(tokenizer.encode(prompt))
    print(f"Tokenized prompt: {token}")

    mx.eval(token)

    # Run the model once with the initial prompt to get the first set of logits.
    logits, _ = model(token[None])
    print(f"Generated logits: {logits}") # Outputs all the Tokens in the vocab size along with the coresponding value that is the most likely next token
    print(f"Size of the generated logits: {logits.size} = vocab_size")
    # Use argmax to select the most likely next token.
    y = mx.argmax(logits[:, -1, :], axis=-1)
    print(f"Selected most likely Token: {y}")
    # Return the first generated token.

    decoded_token = tokenizer.decode([y.item()])
    print(f"Decoded Token: {decoded_token}")
    return y

def generate_after_training(model, prompt, tokenizer, temp, max_tokens):
    print(prompt, end="", flush=True)

    prompt = mx.array(tokenizer.encode(prompt))

    tokens = []
    skip = 0
    for token, n in zip(
        generate(prompt, model, temp),
        range(max_tokens),
    ):
        if token == tokenizer.eos_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        if len(s) - skip > 1:
            print(s[skip:-1], end="", flush=True)
            skip = len(s) - 1
    print(tokenizer.decode(tokens)[skip:], flush=True)
    print("=" * 10)
    if len(tokens) == 0:
        print("No tokens generated for this prompt")
        return



########################################################################################################################
###################### Creating and saving the Model ###################################################################
########################################################################################################################

print("Creating Tokenizer and Model")
model = Model(ModelArgs) # Cerate the Model
# print(model) # Print the Model and it's Architecture



########################################################################################################################
###################### Handling the Datasets ###########################################################################
########################################################################################################################

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
            print(f"Unable to build dataset {dataset_path} ({e})")
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



########################################################################################################################
###################### Training Phase ##################################################################################
########################################################################################################################

# The loss function
def loss(model, inputs, targets, lengths):
    # Run model on inputs
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    # Mask padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the loss
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks

def iterate_batches(dset, tokenizer, batch_size, train=False):
    # Shuffle indices
    while True:
        indices = np.arange(len(dset))
        if train:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            batch = [tokenizer.encode(dset[indices[i + j]]) for j in range(batch_size)]
            lengths = [len(x) for x in batch]

            # Check if any sequence is longer than 2048 tokens
            if max(lengths) > 2048:
                print("[WARNING] Some sequences are longer than 2048 tokens. Consider pre-splitting your data to save memory.")

            # Pad to the max length
            batch_arr = np.zeros((batch_size, max(lengths)), np.int32)

            for j in range(batch_size):
                batch_arr[j, : lengths[j]] = batch[j]
            batch = mx.array(batch_arr)
            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train:
            break

# The evaluation method for getting the loss
def evaluate(model, dataset, loss, tokenizer, batch_size, num_batches):
    all_losses = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(dataset, tokenizer, batch_size)):
        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens

# The Main Trainig loop
def train(model, train_set, val_set, optimizer, loss, tokenizer, args):
    """
        Train the model and plot training and validation losses.

        Args:
            model: The model to train.
            train_set: Dataset used for training.
            val_set: Dataset used for validation.
            optimizer: Optimizer for updating model weights.
            loss: Loss function.
            tokenizer: Tokenizer for data preprocessing.
            args: Training arguments (e.g., batch size, number of iterations).
    """

    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    # Lists to store losses for plotting
    val_losses = []
    iters = []

    train_losses = []
    n_tokens = 0

    # Main training loop
    start = time.perf_counter()

    for it, batch in zip(
        range(args.iters),
        iterate_batches(train_set, tokenizer, args.batch_size, train=True)):

        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        # Model update
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, lvalue)

        # Record loss
        train_losses.append(lvalue.item())
        n_tokens += toks.item()

        # Report training loss if needed
        if (it + 1) % args.steps_per_report == 0:
            train_loss = np.mean(train_losses)

            stop = time.perf_counter()
            print(f"Step {it + 1}: Train loss {train_loss:.3f}, It/sec {args.steps_per_report / (stop - start):.3f}, Tokens/sec {float(n_tokens) / (stop - start):.3f}")
            losses = []
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if it == 0 or (it + 1) % args.steps_per_eval == 0:
            stop = time.perf_counter()
            val_loss = evaluate(
                model, val_set, loss, tokenizer, args.batch_size, args.val_batches
            )
            val_losses.append(val_loss)
            iters.append(it + 1)
            print(f"Step {it + 1}: Validation loss {val_loss:.3f}, Validation time {(time.perf_counter() - stop):.3f}s")

            start = time.perf_counter()

        # Save adapter weights if needed
        if (it + 1) % args.save_every == 0:
            mx.savez(
                args.adapter_file, **dict(tree_flatten(model.trainable_parameters()))
            )
            print(f"Step {it + 1}: Saved adapter weights to {args.adapter_file}.")

        # Plotting the training and validation losses
        plt.figure(figsize=(10, 6))
        plt.plot(iters, val_losses, label='Validation Loss')
        train_loss_points = [train_losses[i - 1] for i in iters]
        plt.plot(iters, train_loss_points, label='Training Loss', alpha=0.5)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        plt.savefig('training_validation_losses.png', dpi=300)
        plt.clf()
        plt.pyplot.close()

###################### LoRA if wanted ####################################################################################

if PreTrainArgs.lora_layers > 0: # Add Low Rank Adaptation for effictient Training
    # Freeze the Model
    model.freeze()

    # select the LoRA Layers and unfrese the model
    for l in model.model.layers[len(model.model.layers) - PreTrainArgs.lora_layers :]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)

        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")

    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")

else:
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Traing Full model with {p:.3f}M parameters")

# Resume training to the given adapters. If exists
if PreTrainArgs.resume_adapter_file is not None:
    print(f"Loading pretrained adapters from {PreTrainArgs.resume_adapter_file}")
    model.load_weights(PreTrainArgs.resume_adapter_file, strict=False)

###################### Start Training ##################################################################################

# Creqate the Optimizer
opt = optim.AdamW(learning_rate=PreTrainArgs.learning_rate, weight_decay=PreTrainArgs.weight_decay)

# Train Model
train(model, train_set, valid_set, opt, loss, tokenizer, PreTrainArgs)

# Save adapter weights
mx.savez(PreTrainArgs.adapter_file, **dict(tree_flatten(model.trainable_parameters())))

if PreTrainArgs.test == True:
    print("Testing that Mofo")
    model.eval()

    test_loss = evaluate(model, test_set, loss, tokenizer, PreTrainArgs.batch_size, num_batches=PreTrainArgs.test_batches)
    test_ppl = math.exp(test_loss)

    print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")



########################################################################################################################
###################### Example Generation after pretraining ############################################################
########################################################################################################################

print("Generating")
prompt = "Hello W"
generate_after_training(model, prompt, tokenizer, temp=0.0, max_tokens=10)



########################################################################################################################
###################### Fine-Tuning Fase ################################################################################
########################################################################################################################

###################### LoRA if wanted ##################################################################################

if FineTuneArgs.lora_layers > 0: # Add Low Rank Adaptation for effictient Training
    # Freeze the Model
    model.freeze()

    # select the LoRA Layers and unfrese the model
    for l in model.model.layers[len(model.model.layers) - FineTuneArgs.lora_layers :]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)

        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")

    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")

else:
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Alligning Full model with {p:.3f}M parameters")

# Resume training to the given adapters. If exists
if FineTuneArgs.resume_adapter_file is not None:
    print(f"Loading pretrained adapters from {FineTuneArgs.resume_adapter_file}")
    model.load_weights(FineTuneArgs.resume_adapter_file, strict=False)

###################### Start Training......Again ########################################################################

# Creqate the Optimizer
opt = optim.AdamW(learning_rate=FineTuneArgs.learning_rate, weight_decay=FineTuneArgs.weight_decay)

# Train Model
train(model, train_set, valid_set, opt, loss, tokenizer, FineTuneArgs)

# Save adapter weights
mx.savez(FineTuneArgs.adapter_file, **dict(tree_flatten(model.trainable_parameters())))

if FineTuneArgs.test == True:
    print("Testing that Mofo")
    model.eval()

    test_loss = evaluate(model, test_set, loss, tokenizer, FineTuneArgs.batch_size, num_batches=FineTuneArgs.test_batches)
    test_ppl = math.exp(test_loss)

    print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")


# And Boom We're done!!!!!



########################################################################################################################
###################### Example Generation after finetuning ############################################################
########################################################################################################################

print("Generating")
prompt = "Hello how are you "
generate_after_training(model, prompt, tokenizer, temp=0.0, max_tokens=10)
