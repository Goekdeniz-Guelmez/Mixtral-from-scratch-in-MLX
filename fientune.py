from create_model import create_model

from add_LoRA import LoRALinear
from mlx.utils import tree_flatten

model_load_path = "/Users/gokdenizgulmez/Desktop/mixtral_mlx/v3"
pretrained_model_save_path = "/Users/gokdenizgulmez/Desktop/mixtral_mlx/v3/pretrained_model.npz"

lora_layers = 16

model, tokenizer = create_model(model_load_path)
print(model)

# Freeze the Model
model.freeze()

# select the LoRA Layers and unfrese the model
for l in model.model.layers[len(model.model.layers) - lora_layers :]:
    l.attention.q_proj = LoRALinear.from_linear(l.attention.q_proj)
    l.attention.v_proj = LoRALinear.from_linear(l.attention.v_proj)

    if hasattr(l, "block_sparse_moe"):
        l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
print(f"Total parameters {p:.3f}M")

p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
print(f"Trainable parameters {p:.3f}M")
