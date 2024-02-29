import time
import wandb
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_flatten

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

def iterate_batches(dset, tokenizer, batch_size, max_seq_length, train=False):
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
            max_length_in_batch = min(max(lengths), max_seq_length)
            batch_arr = np.zeros((batch_size, max_length_in_batch), np.int32)

            for j in range(batch_size):

                truncated_length = min(lengths[j], max_seq_length)

                batch_arr[j, :truncated_length] = batch[j][:truncated_length]

                lengths[j] = (truncated_length)  # Update lengths to match truncated lengths

            batch = mx.array(batch_arr)

        if not train:
            break

# The evaluation method for getting the loss
def evaluate(model, dataset, loss, tokenizer, max_seq_length, batch_size, num_batches):
    all_losses = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(dataset, tokenizer, max_seq_length, batch_size)):
        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens


def train(
    model,
    train_set,
    val_set,
    optimizer,
    tokenizer,
    # report_to_wandb,
    args, loss: callable = loss,
    iterate_batches: callable = iterate_batches
    ):
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
        iterate_batches(dset=train_set, tokenizer=tokenizer, batch_size=args.batch_size, max_seq_length=args.max_position_embeddings, train=True)):

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
            val_loss = evaluate(model=model, dataset=val_set, loss=loss, tokenizer=tokenizer, batch_size=args.batch_size, num_batches=args.val_batches, max_seq_length=args.max_position_embeddings)
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
        plt.close()
