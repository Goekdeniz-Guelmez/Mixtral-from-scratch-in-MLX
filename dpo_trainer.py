import time
import wandb
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_flatten

def DPOLoss(
    beta: float, # Scaling factor for the loss calculation
    policy_chosen_logps: mx.array,
    policy_rejected_logps: mx.array,
    reference_chosen_logps: mx.array,
    reference_rejected_logps: mx.array,
    label_smoothing: float = 0.0, # Label smoothing value for regularization
    ipo: bool = False, # Flag to enable Implicit Preference Optimization
    ) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Implements Direct Preference Optimization (DPO) Loss as a PyTorch module.
    This loss function is used for optimizing policy preferences directly against a reference policy.

    Args:
        beta (float): A scaling factor that adjusts the steepness of the loss function.
        label_smoothing (float, optional): Applies label smoothing for regularization, default is 0.0 (no smoothing).
        ipo (bool, optional): A flag to switch between IPO (Implicit Preference Optimization) and standard DPO, default is False.
        policy_chosen_logps (torch.Tensor): Log probabilities of chosen actions under the policy.
        policy_rejected_logps (torch.Tensor): Log probabilities of rejected actions under the policy.
        reference_chosen_logps (torch.Tensor): Log probabilities of chosen actions under the reference policy.
        reference_rejected_logps (torch.Tensor): Log probabilities of rejected actions under the reference policy.

        Returns:
            Tuple[mx.array, mx.array, mx.array]: A tuple containing the total loss, chosen rewards, and rejected rewards.
        """

    # Calculate log ratios for policy and reference choices
    pi_logratios = policy_chosen_logps - policy_rejected_logps # Log ratio of chosen over rejected actions for the policy
    ref_logratios = reference_chosen_logps - reference_rejected_logps # Log ratio of chosen over rejected actions for the reference
    logits = pi_logratios - ref_logratios # Difference in log ratios between policy and reference
 
    if ipo:
        # Implicit Preference Optimization loss calculation
        losses = (logits - 1 / (2 * beta)) ** 2  # Squared error loss with adjusted logits for IPO. Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        # Direct Preference Optimization loss with optional label smoothing
        losses = (
            - mx.logsigmoid(beta * logits) * (1 - label_smoothing) # Loss for chosen actions
            - mx.logsigmoid(beta * logits) * label_smoothing # Loss for rejected actions with label smoothing
        )

    loss = losses.mean() # Mean loss over all instances
    # Detach chosen and rejected rewards to prevent gradients from flowing through them
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach() # Scaled difference in log probs for chosen actions
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach() # Scaled difference in log probs for rejected actions

    return loss, chosen_rewards, rejected_rewards # Return the computed loss and rewards
