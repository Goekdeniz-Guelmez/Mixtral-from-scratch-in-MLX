# **Mixtral from scratch**  An implementation of Mistral.ai's Mixtral in Apples MLX Framework: Pretrained Mixtral on your MacBook

**Introduction:**

This repository houses an exciting implementation of Mixtral, a state-of-the-art machine learning model leveraging the Sparse Mixture of Experts (MoE) architecture, seamlessly integrated with Apple's MLX framework for lightning-fast inference and training on your MacBook. Apples MLX framework is specifically designed and optimized for Apple Silicon.

This is based on the MLX-Examples repo ffrom the wonderfull Apple research team.

**What is Mixtral?**

Think of Mixtral as a team of specialized AI experts huddled together, each adept at tackling specific parts of complex tasks. By harnessing the combined power of these experts, Mixtral delivers outstanding performance across diverse domains.

!(MoE)[/mixtral_mlx/explainations/mixtral_overview.png]

For more information and a detailed explaination of the Mixtral/Mistral Model, visit the Youtube video by ...

[Mixtral / Mistral explained](https://youtu.be/UiX8K-xBUpE?si=bKlUWhjl0_lJEsjF)

**Key Features:**

- **Sparse Mixture of Experts (Sparse MoE):** Utilizes a dynamic routing mechanism to selectively engage different subsets of parameters (experts) for different inputs, enhancing model capacity and efficiency.
[MoE Paper explained](https://youtu.be/mwO6v4BlgZQ?si=3uVt8Atrk_JvVlAQ)

- **RMS Norm Regularization (Root Mean Square Layer Normalization):** A variant of layer normalization that stabilizes the training of deep networks by normalizing input features based on their RMS value. Ensures smooth collaboration between experts, enhancing overall model stability and accuracy.

[RMS Paper explained](https://youtu.be/mwO6v4BlgZQ?si=3uVt8Atrk_JvVlAQ)

- **Rotary Positional Embedding (RoPE):** Optimizes efficiency by dynamically selecting relevant experts for each input, leading to lower memory requirements and faster inference.

[RoPE Paper explained](https://youtu.be/mwO6v4BlgZQ?si=3uVt8Atrk_JvVlAQ)

- **Low-Rank Adaptation (LoRA):** Adapts the models with minimal additional parameters, focusing on refining the weights of the attention and feedforward networks to improve performance on specific tasks.

**Prerequisites:**

- A MacBook equipped with an Apple M1 chip or later
- Python 3.7+

**Installation:**

1. Clone this repository: `git clone https://github.com/Goekdeniz-Guelmez/train-Mixtral-from-scratch-in-mlx.git`
2. Navigate into the project directory:: `cd mixtral_mlx`
2. Install essential dependencies: `pip install -r requirements.txt`

**Project Structure:**

- config.json: The configuration file for the Model to create.
- Mixtral.py: Mixtral model architecture.
- tokenizer.model: The Tokenizer.
- train_that_mf.py: Training and evaluation loops, optimizer setup.
- add_lora.py: Low-Rank Adaptation implementation for efficient Finetuning.
- load_file.py: Loads the Model weights.
- create_model.py: Creates the Mixtral Model, based in the configuration json file.
- pretrain_data/: Data sets for Pre-training, validation, and testing.
- finetune_data/: Data sets for Fine-tuning, validation, and testing.
- pretrain.py: The File for Pretraineing the Model.
- Finetunine.py: The File for Finetuning the Model.

**Pretraining:**

For pretraining, you will need a `data` folder containing three .jsonl files: `train.jsonl`, `val.jsonl`, and `test.jsonl`. Each file should consist of multiple lines, where each line is a JSON object with a `text` key containing the training text as a string. This repo contains the wikitext-2-raw-v1 dataset.

After that, you can Pretraine the mixtral Model with:

```sh
python pretrain.py
```

**Finetuning:**

The same dataset structure goes for the Fintuning fase. with the conversational text pairs in the format you want.

```sh
python finetune.py
```

**Example of a Pre-training set**:
```json
{"text": "Your training text goes here."}
{"text": "Your training text goes here."}
...
```

**Example of a Fine-tuning set**:
```json
{"text": "###user: {} ###Assistant..."}
{"text": "###user: {} ###Assistant..."}
...
```

**Running Inference:**

There are 4 diferent geenration methods in the code:

1. `generate:` Generates text based on the given prompt and temperature.
2. `generate_raw:` Generates the raw logits on the tokenized prompt.
3. `generate_one_token:` Generates a single text token based on the given prompt and tmperature.
4. `just_generate_one_god_damn_token:` Generates a single text token based on the given prompt, using the most likely prediction. This is more for testing.
5. `generate_after_training`: Do I need to tell you what is does???????

**Beyond the Basics:**

For advanced customization and exploring Mixtral's full potential, dive into the codebase and reference the provided documentation.

**Resources:**

- Mistral AI website: [https://mistral.ai/](https://mistral.ai/)
- Apple MLX framework Examples: [https://github.com/ml-explore/mlx-examples/tree/main]

**Contributing:**

We welcome your contributions! Please feel free to open pull requests or reach out with suggestions or questions.

**Remember to replace `<path/to/model>` with the actual location of your model and personalize the example output according to your project.**

I believe this refined README effectively blends the strengths of both responses, providing a clear, concise, and engaging introduction to your Mixtral implementation while addressing the specific requirements of the task.
