import torch
import optuna
import dill
from transformers import AutoModel, AutoTokenizer
from copy import deepcopy
from pathlib import Path
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr
from chop.nn.quantized.modules.linear import LinearInteger
from optuna.samplers import GridSampler, RandomSampler, TPESampler
import matplotlib.pyplot as plt
import numpy as np

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

# Load the best model from Lab 2 Task 1
model_path = Path(__file__).parent.parent / "Lab_2" / "Lab_2_Task_1_Best_Model.pkl"
with open(model_path, "rb") as f:
    base_model = dill.load(f)

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

search_space = {
    "linear_layer_choices": [
        torch.nn.Linear,
        LinearInteger,
    ],
    # Per-layer quantization options
    "width_choices": [8, 16, 32],           # Bit widths
    "frac_width_choices": [2, 4, 8],        # Fractional bit widths
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
            
            # Choose quantization parameters for this specific layer
            # Different layers can now have different quantisation
            data_width = trial.suggest_categorical(
                f"{name}_data_width",
                search_space["width_choices"],
            )
            
            weight_width = trial.suggest_categorical(
                f"{name}_weight_width",
                search_space["width_choices"],
            )

            bias_width = trial.suggest_categorical(
                f"{name}_bias_width",
                search_space["width_choices"],
            )
            
            data_frac = trial.suggest_categorical(
                f"{name}_data_frac",
                search_space["frac_width_choices"],
            )
            
            weight_frac = trial.suggest_categorical(
                f"{name}_weight_frac",
                search_space["frac_width_choices"],
            )

            bias_frac = trial.suggest_categorical(
                f"{name}_bias_frac",
                search_space["frac_width_choices"],
            )
            
            # Ensure fractional width doesn't exceed total width
            # (frac_width should be < width for valid fixed-point)
            data_frac = min(data_frac, data_width - 1)
            weight_frac = min(weight_frac, weight_width - 1)
            bias_frac = min(bias_frac, bias_width - 1)

            kwargs = {
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
            }

            # Create quantized layer with per-layer config
            if new_layer_cls == LinearInteger:
                kwargs["config"] = {
                        "data_in_width": data_width,
                        "data_in_frac_width": data_frac,
                        "weight_width": weight_width,
                        "weight_frac_width": weight_frac,
                        "bias_width": bias_width,
                        "bias_frac_width": bias_frac,
                }
            
            # Create new layer and copy weights
            new_layer = new_layer_cls(**kwargs)
            new_layer.weight.data = layer.weight.data
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data
            
            # Replace the layer
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
    accuracy = eval_results["eval_accuracy"]
    
    print(f"Trial {trial.number}: Accuracy = {accuracy:.4f}")
    
    # Save model for later retrieval
    trial.set_user_attr("model", model)
    
    return accuracy

sampler = TPESampler()

n_trials = 20

study = optuna.create_study(
    direction="maximize",
    study_name="mixed-precision-quantization",
    sampler=sampler,
)

study.optimize(
    objective,
    n_trials=n_trials,
)

def get_cumulative_max(study):
    values = [t.value for t in study.trials if t.value is not None]
    return np.maximum.accumulate(values)

cumulative_max = get_cumulative_max(study)
trial_numbers = list(range(1, len(cumulative_max) + 1))

plt.figure(figsize=(12, 7))

plt.plot(
    trial_numbers,
    cumulative_max,
    marker='o',
    label='Mixed-Precision Quantization',
    color='#E63946',
    linewidth=2.5,
    markersize=7,
    alpha=0.9
)

plt.xlabel("Number of Trials", fontsize=14, fontweight='bold')
plt.ylabel("Maximum Achieved Accuracy", fontsize=14, fontweight='bold')
plt.title("Mixed-Precision Quantization Search",
          fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=11, loc='lower right', framealpha=0.95, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save figure
save_path = Path(__file__).parent / "Lab_3_Mixed_Precision_Search.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()


print(f"\nBest Trial:")
print(f"  Accuracy: {study.best_value:.4f}")
print(f"  Trial Number: {study.best_trial.number}")
print(f"  Total Trials: {len(study.trials)}")

best_params = study.best_trial.params

# Group parameters by layer
layer_configs = {}
for param_name, param_value in best_params.items():
    if "_type" in param_name:
        layer_name = param_name.replace("_type", "")
        if param_value == LinearInteger:
            layer_configs[layer_name] = {
                "type": "LinearInteger",
                "data_width": best_params.get(f"{layer_name}_data_width", "N/A"),
                "data_frac": best_params.get(f"{layer_name}_data_frac", "N/A"),
                "weight_width": best_params.get(f"{layer_name}_weight_width", "N/A"),
                "weight_frac": best_params.get(f"{layer_name}_weight_frac", "N/A"),
                "bias_width": best_params.get(f"{layer_name}_bias_width", "N/A"),
                "bias_frac": best_params.get(f"{layer_name}_bias_frac", "N/A"),
            }
        else:
            layer_configs[layer_name] = {"type": "Linear (FP32)"}

# Analyze bitwidth distribution
quantized_layers = [c for c in layer_configs.values() if c["type"] == "LinearInteger"]
print("QUANTIZATION:")
print(f"Total Layers: {len(layer_configs)}")
print(f"Quantized Layers: {len(quantized_layers)}")
print(f"FP32 Layers: {len(layer_configs) - len(quantized_layers)}")

if quantized_layers:
    data_widths = [c["data_width"] for c in quantized_layers]
    weight_widths = [c["weight_width"] for c in quantized_layers]
    bias_widths = [c["bias_width"] for c in quantized_layers]
    
    print(f"\nData Width Distribution:")
    for width in [8, 16, 32]:
        count = data_widths.count(width)
        if count > 0:
            print(f"  {width}-bit: {count} layers ({count/len(quantized_layers)*100:.1f}%)")
    
    print(f"\nWeight Width Distribution:")
    for width in [8, 16, 32]:
        count = weight_widths.count(width)
        if count > 0:
            print(f"  {width}-bit: {count} layers ({count/len(quantized_layers)*100:.1f}%)")

    print(f"\nBias Width Distribution:")
    for width in [8, 16, 32]:
        count = bias_widths.count(width)
        if count > 0:
            print(f"  {width}-bit: {count} layers ({count/len(quantized_layers)*100:.1f}%)")
    
    avg_data = sum(data_widths) / len(data_widths)
    avg_weight = sum(weight_widths) / len(weight_widths)
    avg_bias = sum(bias_widths) / len(bias_widths)

# Save best model
best_model = study.best_trial.user_attrs["model"].cpu()
model_path = Path(__file__).parent / "Lab_3_Best_Mixed_Precision_Model.pkl"
with open(model_path, "wb") as f:
    dill.dump(best_model, f)

# Best Trial:
#  Accuracy: 0.8746
#  Trial Number: 14
#  Total Trials: 20

# QUANTIZATION:
# Total Layers: 14
# Quantized Layers: 6
# FP32 Layers: 8

# Data Width Distribution:
#  8-bit: 3 layers (50.0%)
#  16-bit: 2 layers (33.3%)
#  32-bit: 1 layers (16.7%)

# Weight Width Distribution:
#  8-bit: 5 layers (83.3%)
#  16-bit: 1 layers (16.7%)

# Bias Width Distribution:
#  8-bit: 1 layers (16.7%)
#  16-bit: 3 layers (50.0%)
#  32-bit: 2 layers (33.3%)