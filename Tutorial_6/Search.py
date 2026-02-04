import torch
import optuna
import dill
from transformers import AutoModel, AutoTokenizer
from copy import deepcopy
from pathlib import Path
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr
from chop.nn.quantized.modules.linear import (
    LinearInteger,
    LinearMinifloatDenorm,
    LinearMinifloatIEEE,
    LinearLog,
    LinearBlockFP,
    LinearBlockMinifloat,
    LinearBlockLog,
    LinearBinary,
    LinearBinaryScaling,
    LinearBinaryResidualSign,
)
from optuna.samplers import GridSampler, RandomSampler, TPESampler

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

model_path = Path(__file__).parent.parent / "Tutorial_5" / "Tutorial_5_Best_Model.pkl"
with open(model_path, "rb") as f:
    base_model = dill.load(f)

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

# Layers which Optuna can choose from
search_space = {
    "linear_layer_choices": [
        torch.nn.Linear,
        LinearInteger,
    ],
}

def construct_model(trial):
    # Creates a fresh copy of the base model
    trial_model = deepcopy(base_model)

    # Iterate through all modules of the model
    for name, layer in trial_model.named_modules():
        # Only care about replacing Linear Layers
        if isinstance(layer, torch.nn.Linear):
            # Ask optuna to pick a layer type for this specific layer name
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )

            # If it picks standard Linear, do nothing and continue
            if new_layer_cls == torch.nn.Linear:
                continue

            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
            }

            # If the chosen layer is integer, define the low precision config
            if new_layer_cls == LinearInteger:
                kwargs["config"] = {
                    "data_in_width": 8,
                    "data_in_frac_width": 4,
                    "weight_width": 8,
                    "weight_frac_width": 6,
                    "bias_width": 8,
                    "bias_frac_width": 6,
                }
            # elif... (other precisions)

            # Create the new layer (copy the weights from the original one)
            new_layer = new_layer_cls(**kwargs)
            new_layer.weight.data = layer.weight.data

            # Replace the layer in the model
            deepsetattr(trial_model, name, new_layer)

    return trial_model

def objective(trial):
    # Define the model
    model = construct_model(trial)

    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    # Save the model to the trial so we can retrieve it later if it's the best
    trial.set_user_attr("model", model)

    # Optuna will try to MAXIMIZE this value
    return eval_results["eval_accuracy"]

# Can change this to GridSampler or TPESampler
sampler = RandomSampler()

study = optuna.create_study(
    direction="maximize",
    study_name="bert-tiny-nas-study",
    sampler=sampler,
)

study.optimize(
    objective,
    n_trials=1,
    timeout=60 * 60 * 24,
)

print(f"Best trial accuracy: {study.best_value}")
print(f"Best parameters: {study.best_params}")

# Best trial accuracy: 0.86084