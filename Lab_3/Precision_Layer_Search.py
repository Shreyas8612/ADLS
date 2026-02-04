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
)
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


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

# Extended search space with all precision types
search_space = {
    "linear_layer_choices": [
        torch.nn.Linear,          # Full precision FP32
        LinearInteger,            # Fixed-point quantization
        LinearMinifloatDenorm,    # Minifloat with denormal support
        LinearMinifloatIEEE,      # Minifloat IEEE format
        LinearLog,                # Logarithmic quantization
        LinearBlockFP,            # Block floating-point
        LinearBlockMinifloat,     # Block minifloat
        LinearBlockLog,           # Block logarithmic
        LinearBinary,             # Binary quantization
    ],
    # Hyperparameter ranges
    "width_choices": [8, 16, 32],
    "frac_width_choices": [2, 4, 8],
    "exponent_width_choices": [3, 4, 5],
    "exponent_bias_choices": [3, 7, 15],
    "exponent_bias_width_choices": [4, 6, 8],
    "block_size_choices": [32, 64, 128],
}


def construct_model(trial):
    trial_model = deepcopy(base_model)
    
    for name, layer in trial_model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            # Ask Optuna to pick a layer type for this specific layer
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )
            
            # If it picks standard Linear, do nothing and continue
            if new_layer_cls == torch.nn.Linear:
                continue
            
            # Base kwargs for all layer types
            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
            }
            
            if new_layer_cls == LinearInteger:
                # Search over width and fractional width for each component
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
                data_frac = min(data_frac, data_width - 1)
                weight_frac = min(weight_frac, weight_width - 1)
                bias_frac = min(bias_frac, bias_width - 1)
                
                kwargs["config"] = {
                    "data_in_width": data_width,
                    "data_in_frac_width": data_frac,
                    "weight_width": weight_width,
                    "weight_frac_width": weight_frac,
                    "bias_width": bias_width,
                    "bias_frac_width": bias_frac,
                }
            
            elif new_layer_cls in [LinearMinifloatDenorm, LinearMinifloatIEEE]:
                # Required: width, exponent_width, exponent_bias for data_in, weight, bias
                data_width = trial.suggest_categorical(f"{name}_data_width", search_space["width_choices"])
                weight_width = trial.suggest_categorical(f"{name}_weight_width", search_space["width_choices"])
                bias_width = trial.suggest_categorical(f"{name}_bias_width", search_space["width_choices"])
                data_exp_width = trial.suggest_categorical(f"{name}_data_exp_width", search_space["exponent_width_choices"])
                weight_exp_width = trial.suggest_categorical(f"{name}_weight_exp_width", search_space["exponent_width_choices"])
                bias_exp_width = trial.suggest_categorical(f"{name}_bias_exp_width", search_space["exponent_width_choices"])
                data_exp_bias = trial.suggest_categorical(f"{name}_data_exp_bias", search_space["exponent_bias_choices"])
                weight_exp_bias = trial.suggest_categorical(f"{name}_weight_exp_bias", search_space["exponent_bias_choices"])
                bias_exp_bias = trial.suggest_categorical(f"{name}_bias_exp_bias", search_space["exponent_bias_choices"])
                
                kwargs["config"] = {
                    "data_in_width": data_width,
                    "data_in_exponent_width": data_exp_width,
                    "data_in_exponent_bias": data_exp_bias,
                    "weight_width": weight_width,
                    "weight_exponent_width": weight_exp_width,
                    "weight_exponent_bias": weight_exp_bias,
                    "bias_width": bias_width,
                    "bias_exponent_width": bias_exp_width,
                    "bias_exponent_bias": bias_exp_bias,
                }
            
            elif new_layer_cls == LinearLog:
                # Required: width, exponent_bias for data_in, weight, bias (no exponent_width)
                data_width = trial.suggest_categorical(f"{name}_data_width", search_space["width_choices"])
                weight_width = trial.suggest_categorical(f"{name}_weight_width", search_space["width_choices"])
                bias_width = trial.suggest_categorical(f"{name}_bias_width", search_space["width_choices"])
                data_exp_bias = trial.suggest_categorical(f"{name}_data_exp_bias", search_space["exponent_bias_choices"])
                weight_exp_bias = trial.suggest_categorical(f"{name}_weight_exp_bias", search_space["exponent_bias_choices"])
                bias_exp_bias = trial.suggest_categorical(f"{name}_bias_exp_bias", search_space["exponent_bias_choices"])
                
                kwargs["config"] = {
                    "data_in_width": data_width,
                    "data_in_exponent_bias": data_exp_bias,
                    "weight_width": weight_width,
                    "weight_exponent_bias": weight_exp_bias,
                    "bias_width": bias_width,
                    "bias_exponent_bias": bias_exp_bias,
                }
            
            elif new_layer_cls == LinearBlockFP:
                # Required: width, exponent_width, exponent_bias, block_size for data_in, weight, bias
                data_width = trial.suggest_categorical(f"{name}_data_width", search_space["width_choices"])
                weight_width = trial.suggest_categorical(f"{name}_weight_width", search_space["width_choices"])
                bias_width = trial.suggest_categorical(f"{name}_bias_width", search_space["width_choices"])
                data_exp_width = trial.suggest_categorical(f"{name}_data_exp_width", search_space["exponent_width_choices"])
                weight_exp_width = trial.suggest_categorical(f"{name}_weight_exp_width", search_space["exponent_width_choices"])
                bias_exp_width = trial.suggest_categorical(f"{name}_bias_exp_width", search_space["exponent_width_choices"])
                data_exp_bias = trial.suggest_categorical(f"{name}_data_exp_bias", search_space["exponent_bias_choices"])
                weight_exp_bias = trial.suggest_categorical(f"{name}_weight_exp_bias", search_space["exponent_bias_choices"])
                bias_exp_bias = trial.suggest_categorical(f"{name}_bias_exp_bias", search_space["exponent_bias_choices"])
                block_size = trial.suggest_categorical(f"{name}_block_size", search_space["block_size_choices"])
                
                kwargs["config"] = {
                    "data_in_width": data_width,
                    "data_in_exponent_width": data_exp_width,
                    "data_in_exponent_bias": data_exp_bias,
                    "data_in_block_size": [block_size],
                    "weight_width": weight_width,
                    "weight_exponent_width": weight_exp_width,
                    "weight_exponent_bias": weight_exp_bias,
                    "weight_block_size": [block_size],
                    "bias_width": bias_width,
                    "bias_exponent_width": bias_exp_width,
                    "bias_exponent_bias": bias_exp_bias,
                    "bias_block_size": [block_size],
                }

            elif new_layer_cls == LinearBlockMinifloat:
                # Required: width, exponent_width, exponent_bias_width, block_size for data_in, weight, bias
                data_width = trial.suggest_categorical(f"{name}_data_width", search_space["width_choices"])
                weight_width = trial.suggest_categorical(f"{name}_weight_width", search_space["width_choices"])
                bias_width = trial.suggest_categorical(f"{name}_bias_width", search_space["width_choices"])
                data_exp_width = trial.suggest_categorical(f"{name}_data_exp_width", search_space["exponent_width_choices"])
                weight_exp_width = trial.suggest_categorical(f"{name}_weight_exp_width", search_space["exponent_width_choices"])
                bias_exp_width = trial.suggest_categorical(f"{name}_bias_exp_width", search_space["exponent_width_choices"])
                data_exp_bias_width = trial.suggest_categorical(f"{name}_data_exp_bias_width", search_space["exponent_bias_width_choices"])
                weight_exp_bias_width = trial.suggest_categorical(f"{name}_weight_exp_bias_width", search_space["exponent_bias_width_choices"])
                bias_exp_bias_width = trial.suggest_categorical(f"{name}_bias_exp_bias_width", search_space["exponent_bias_width_choices"])
                block_size = trial.suggest_categorical(f"{name}_block_size", search_space["block_size_choices"])
                
                kwargs["config"] = {
                    "data_in_width": data_width,
                    "data_in_exponent_width": data_exp_width,
                    "data_in_exponent_bias_width": data_exp_bias_width,
                    "data_in_block_size": [block_size],
                    "weight_width": weight_width,
                    "weight_exponent_width": weight_exp_width,
                    "weight_exponent_bias_width": weight_exp_bias_width,
                    "weight_block_size": [block_size],
                    "bias_width": bias_width,
                    "bias_exponent_width": bias_exp_width,
                    "bias_exponent_bias_width": bias_exp_bias_width,
                    "bias_block_size": [block_size],
                }
            
            elif new_layer_cls == LinearBlockLog:
                # Required: width, exponent_bias_width, block_size for data_in, weight, bias
                data_width = trial.suggest_categorical(f"{name}_data_width", search_space["width_choices"])
                weight_width = trial.suggest_categorical(f"{name}_weight_width", search_space["width_choices"])
                bias_width = trial.suggest_categorical(f"{name}_bias_width", search_space["width_choices"])
                data_exp_bias_width = trial.suggest_categorical(f"{name}_data_exp_bias_width", search_space["exponent_bias_width_choices"])
                weight_exp_bias_width = trial.suggest_categorical(f"{name}_weight_exp_bias_width", search_space["exponent_bias_width_choices"])
                bias_exp_bias_width = trial.suggest_categorical(f"{name}_bias_exp_bias_width", search_space["exponent_bias_width_choices"])
                block_size = trial.suggest_categorical(f"{name}_block_size", search_space["block_size_choices"])
                
                kwargs["config"] = {
                    "data_in_width": data_width,
                    "data_in_exponent_bias_width": data_exp_bias_width,
                    "data_in_block_size": [block_size],
                    "weight_width": weight_width,
                    "weight_exponent_bias_width": weight_exp_bias_width,
                    "weight_block_size": [block_size],
                    "bias_width": bias_width,
                    "bias_exponent_bias_width": bias_exp_bias_width,
                    "bias_block_size": [block_size],
                }

            elif new_layer_cls == LinearBinary:
                # Required: weight_stochastic, weight_bipolar
                # NOTE: bipolar must be True for Linear layers (2D weights)
                weight_stochastic = trial.suggest_categorical(f"{name}_weight_stochastic", [True, False])
                
                kwargs["config"] = {
                    "weight_stochastic": weight_stochastic,
                    "weight_bipolar": True,  # Must be True for Linear layers
                }
            
            # Create new layer and copy weights
            new_layer = new_layer_cls(**kwargs)
            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None and hasattr(new_layer, 'bias') and new_layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
            
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
    accuracy = eval_results["eval_accuracy"]
    
    print(f"Trial {trial.number}: Accuracy = {accuracy:.4f}")
    
    # Save model for later retrieval
    trial.set_user_attr("model", model)
    
    return accuracy


# Use TPE sampler for efficient search
sampler = TPESampler()

n_trials = 30

study = optuna.create_study(
    direction="maximize",
    study_name="all-precision-search",
    sampler=sampler,
)

study.optimize(
    objective,
    n_trials=n_trials,
)

def get_trials_by_precision(study):
    precision_trials = defaultdict(list)
    
    for trial in study.trials:
        if trial.value is None or trial.value == 0.0:
            continue
        
        # Determine dominant precision type for this trial
        precision_counts = defaultdict(int)
        for param_name, param_value in trial.params.items():
            if "_type" in param_name:
                if param_value == torch.nn.Linear:
                    precision_name = "FP32"
                else:
                    precision_name = param_value.__name__.replace("Linear", "")
                precision_counts[precision_name] += 1
        
        # Use the most common precision type
        if precision_counts:
            dominant_precision = max(precision_counts, key=precision_counts.get)
            max_count = precision_counts[dominant_precision]
            total_layers = sum(precision_counts.values())
            
            # If one precision dominates (>50% of layers), label as that type
            if max_count > total_layers * 0.5:
                label = dominant_precision
            else:
                label = "Mixed"
        else:
            label = "FP32"
        
        precision_trials[label].append((trial.number, trial.value))
    
    return precision_trials

def get_cumulative_max_per_precision(precision_trials):
    precision_cummax = {}
    
    for precision, trials in precision_trials.items():
        # Sort by trial number
        trials_sorted = sorted(trials, key=lambda x: x[0])
        
        if not trials_sorted:
            continue
            
        # Create arrays for plotting
        trial_numbers = []
        cummax_values = []
        current_max = -float('inf')
        
        for trial_num, accuracy in trials_sorted:
            trial_numbers.append(trial_num)
            current_max = max(current_max, accuracy)
            cummax_values.append(current_max)
        
        precision_cummax[precision] = (trial_numbers, cummax_values)
    
    return precision_cummax


# Get trials grouped by precision
precision_trials = get_trials_by_precision(study)
precision_cummax = get_cumulative_max_per_precision(precision_trials)

# Define colors for different precision types
color_map = {
    "FP32": "#2E86AB",
    "Integer": "#A23B72",
    "MinifloatDenorm": "#F18F01",
    "MinifloatIEEE": "#C73E1D",
    "Log": "#6A994E",
    "BlockFP": "#BC4B51",
    "BlockMinifloat": "#8B5A3C",
    "BlockLog": "#5E548E",
    "Binary": "#9A031E",
    "Mixed": "#6C757D",
}

# Plot cumulative maximum accuracy for each precision type
plt.figure(figsize=(14, 8))

for precision, (trial_numbers, cummax) in sorted(precision_cummax.items(), 
                                                  key=lambda x: max(x[1][1]) if x[1][1] else 0, 
                                                  reverse=True):
    color = color_map.get(precision, "#000000")
    marker = 'o'
    
    plt.plot(
        trial_numbers,
        cummax,
        marker=marker,
        label=f"{precision} (best: {max(cummax):.4f})",
        color=color,
        linewidth=2.5,
        markersize=6,
        alpha=0.85,
        markevery=max(1, len(trial_numbers) // 10)  # Show markers at regular intervals
    )

plt.xlabel("Trial Number", fontsize=14, fontweight='bold')
plt.ylabel("Maximum Achieved Accuracy", fontsize=14, fontweight='bold')
plt.title("Precision-Aware NAS: Cumulative Best Accuracy by Quantization Type",
          fontsize=15, fontweight='bold', pad=20)
plt.legend(fontsize=9, loc='lower right', framealpha=0.95, shadow=True, ncol=2)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save figure
save_path = Path(__file__).parent / "Lab_3_Task_2_All_Precision_Search.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()


# ==================== PRINT SUMMARY ====================

print(f"\n{'='*70}")
print(f"{'SEARCH SUMMARY':^70}")
print(f"{'='*70}\n")

print(f"Best Trial:")
print(f"  Accuracy: {study.best_value:.4f}")
print(f"  Trial Number: {study.best_trial.number}")
print(f"  Total Trials: {len(study.trials)}")

print(f"\n{'='*70}")
print(f"{'PRECISION TYPE STATISTICS':^70}")
print(f"{'='*70}\n")

# Count trials by precision type
precision_counts = defaultdict(int)
precision_best_acc = defaultdict(lambda: 0.0)
precision_avg_acc = defaultdict(list)

for precision, trials in precision_trials.items():
    precision_counts[precision] = len(trials)
    if trials:
        precision_best_acc[precision] = max(t[1] for t in trials)
        precision_avg_acc[precision] = [t[1] for t in trials]

print(f"{'Precision Type':<25} {'Trials':<10} {'Best Acc':<12} {'Avg Acc':<12}")
print(f"{'-'*60}")

for precision in sorted(precision_counts.keys(), key=lambda x: precision_best_acc[x], reverse=True):
    count = precision_counts[precision]
    best_acc = precision_best_acc[precision]
    avg_acc = np.mean(precision_avg_acc[precision]) if precision_avg_acc[precision] else 0.0
    print(f"{precision:<25} {count:<10} {best_acc:.4f}      {avg_acc:.4f}")

print(f"\n{'='*70}\n")

# Analyze best trial
best_params = study.best_trial.params
layer_types = {}

for param_name, param_value in best_params.items():
    if "_type" in param_name:
        layer_name = param_name.replace("_type", "")
        if param_value == torch.nn.Linear:
            layer_types[layer_name] = "FP32"
        else:
            layer_types[layer_name] = param_value.__name__

type_distribution = defaultdict(int)
for layer_type in layer_types.values():
    type_distribution[layer_type] += 1

print(f"Best Trial Configuration:")
print(f"  Total Layers: {len(layer_types)}")
for layer_type, count in sorted(type_distribution.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(layer_types)) * 100
    print(f"  {layer_type}: {count} layers ({percentage:.1f}%)")

# Save best model
if study.best_trial.user_attrs.get("model") is not None:
    best_model = study.best_trial.user_attrs["model"].cpu()
    model_path = Path(__file__).parent / "Lab_3_Task_2_Best_All_Precision_Model.pkl"
    with open(model_path, "wb") as f:
        dill.dump(best_model, f)
    print(f"\nBest model saved to: {model_path}")

print(f"Plot saved to: {save_path}")