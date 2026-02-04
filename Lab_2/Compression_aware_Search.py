import torch
import torch.nn as nn
from chop.nn.modules import Identity
from chop.tools import get_tokenized_dataset, get_trainer
from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools.utils import deepsetattr
import optuna
from optuna.samplers import GridSampler
from chop import MaseGraph
import chop.passes as passes
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import dill

# Load dataset
dataset, tokenizer = get_tokenized_dataset(
    dataset="imdb",
    checkpoint="bert-base-uncased",
    return_tokenizer=True,
)

# Define the Search Spac
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choices": [nn.Linear, Identity], 
}

def construct_model_for_grid(trial):
    config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
    
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        options = search_space[param]
        chosen_idx = trial.suggest_int(param, 0, len(options) - 1)
        setattr(config, param, options[chosen_idx])
    
    model = AutoModelForSequenceClassification.from_config(config)
    model.config.problem_type = "single_label_classification"

    layer_type_choice = trial.suggest_int("global_layer_type", 0, 1)
    
    if layer_type_choice == 1:
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
                deepsetattr(model, name, Identity())
    
    return model

def get_quantization_config_8bit():
    return {
        "by": "type",
        "default": {"config": {"name": None}},
        "linear": {
            "config": {
                "name": "integer",
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "weight_width": 8,
                "weight_frac_width": 6,
                "bias_width": 8,
                "bias_frac_width": 6,
            }
        },
    }

def get_pruning_config_l1norm(sparsity=0.5):
    return {
        "weight": {
            "sparsity": sparsity,
            "method": "l1-norm",
            "scope": "local",
        },
        "activation": {
            "sparsity": sparsity,
            "method": "l1-norm",
            "scope": "local",
        },
    }

def apply_compression(model, sparsity=0.5):
    model = model.cpu()
    
    # Convert to MaseGraph directly from the model
    mg = MaseGraph(
        model,
        hf_input_names=[
            "input_ids", 
            "attention_mask", 
            "labels"
        ],
    )
    
    # Initialize metadata analysis
    mg, _ = passes.init_metadata_analysis_pass(mg)
    mg, _ = passes.add_common_metadata_analysis_pass(mg)
    
    # Apply 8-bit quantization
    mg, _ = passes.quantize_transform_pass(
        mg,
        pass_args=get_quantization_config_8bit(),
    )
    
    # Apply L1-norm pruning
    mg, _ = passes.prune_transform_pass(
        mg,
        pass_args=get_pruning_config_l1norm(sparsity=sparsity),
    )
    
    return mg.model

def objective_baseline(trial):
    model = construct_model_for_grid(trial)
    
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,  # Initial training
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    accuracy = eval_results["eval_accuracy"]
    
    print(f"[Baseline] Trial {trial.number}: Accuracy = {accuracy:.4f}")
    trial.set_user_attr("model", model)
    
    return accuracy

def objective_compression_no_post_training(trial):
    # Compression-aware WITHOUT post-compression training
    # Flow: Build -> Train -> Quantize -> Prune -> Evaluate (PTQ)
    model = construct_model_for_grid(trial)
    
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    
    trainer.train()
    
    # Apply compression (Quantization + Pruning)
    try:
        compressed_model = apply_compression(model, sparsity=0.5)
    except Exception as e:
        print(f"[No Post] Trial {trial.number}: Compression failed - {e}")
        return 0.0
    
    # Evaluate immediately (PTQ style - no training)
    trainer = get_trainer(
        model=compressed_model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
    )
    
    eval_results = trainer.evaluate()
    accuracy = eval_results["eval_accuracy"]
    
    print(f"[Compression No Post] Trial {trial.number}: Accuracy = {accuracy:.4f}")
    trial.set_user_attr("model", compressed_model)
    
    return accuracy

def objective_compression_with_post_training(trial):
    # Compression-aware WITH post-compression training
    # Flow: Build -> Train -> Quantize -> Prune -> Fine-tune -> Evaluate (QAT)
    model = construct_model_for_grid(trial)
    
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    
    trainer.train()
    
    # Apply compression (Quantization + Pruning)
    try:
        compressed_model = apply_compression(model, sparsity=0.5)
    except Exception as e:
        print(f"[With Post] Trial {trial.number}: Compression failed - {e}")
        return 0.0
    
    # Fine-tune compressed model (QAT + Pruning fine-tuning)
    trainer = get_trainer(
        model=compressed_model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=3,
    )
    
    trainer.train()
    
    # Final evaluation
    eval_results = trainer.evaluate()
    accuracy = eval_results["eval_accuracy"]
    
    print(f"[Compression With Post] Trial {trial.number}: Accuracy = {accuracy:.4f}")
    trial.set_user_attr("model", compressed_model)
    
    return accuracy

n_trials = 20

grid_sampler_config = {
    "num_layers": [0, 1, 2],
    "num_heads": [0, 1, 2, 3],
    "hidden_size": [0, 1, 2, 3, 4],
    "intermediate_size": [0, 1, 2, 3, 4],
    "global_layer_type": [0, 1]
}

# Baseline (No Compression)
print("\nRunning Baseline (No Compression):")
study_baseline = optuna.create_study(
    direction="maximize",
    study_name="task2-baseline",
    sampler=GridSampler(search_space=grid_sampler_config),
)
study_baseline.optimize(objective_baseline, n_trials=n_trials)

# Compression WITHOUT post-training (PTQ style)
print("\nRunning Compression-Aware WITHOUT Post-Training (PTQ):")
study_no_post = optuna.create_study(
    direction="maximize",
    study_name="task2-compression-ptq",
    sampler=GridSampler(search_space=grid_sampler_config),
)
study_no_post.optimize(objective_compression_no_post_training, n_trials=n_trials)

# Compression WITH post-training (QAT style)
print("\nRunning Compression-Aware WITH Post-Training (QAT + Fine-tuning)")
study_with_post = optuna.create_study(
    direction="maximize",
    study_name="task2-compression-qat",
    sampler=GridSampler(search_space=grid_sampler_config),
)
study_with_post.optimize(objective_compression_with_post_training, n_trials=n_trials)

def get_cumulative_max(study):
    values = [t.value for t in study.trials if t.value is not None]
    return np.maximum.accumulate(values)

results = {
    "No Compression": get_cumulative_max(study_baseline),
    "Compression w/o Post-Training (PTQ)": get_cumulative_max(study_no_post),
    "Compression w/ Post-Training (QAT)": get_cumulative_max(study_with_post),
}

plt.figure(figsize=(12, 7))

colors = {
    "No Compression": "#2E86AB",      # Blue
    "Compression w/o Post-Training (PTQ)": "#E63946",  # Red
    "Compression w/ Post-Training (QAT)": "#06A77D"    # Green
}

markers = {
    "No Compression": "o",
    "Compression w/o Post-Training (PTQ)": "s",
    "Compression w/ Post-Training (QAT)": "^"
}

linestyles = {
    "No Compression": "-",
    "Compression w/o Post-Training (PTQ)": "--",
    "Compression w/ Post-Training (QAT)": "-"
}

for label, cumulative_max in results.items():
    trial_numbers = list(range(1, len(cumulative_max) + 1))
    plt.plot(
        trial_numbers, 
        cumulative_max, 
        marker=markers[label],
        label=label,
        color=colors[label],
        linestyle=linestyles[label],
        linewidth=2.5,
        markersize=7,
        alpha=0.9
    )

plt.xlabel("Number of Trials", fontsize=14, fontweight='bold')
plt.ylabel("Maximum Achieved Accuracy", fontsize=14, fontweight='bold')
plt.title("Compression-Aware NAS Performance Comparison\n", 
          fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=11, loc='lower right', framealpha=0.95, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save figure
save_path = Path(__file__).parent / "Lab_2_Task_2_Compression_Comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

studies = {
    "No Compression": study_baseline,
    "Compression w/o Post-Training (PTQ)": study_no_post,
    "Compression w/ Post-Training (QAT)": study_with_post,
}

for name, study in studies.items():
    print(f"\n{name}:")
    print(f"  Best Accuracy: {study.best_value:.4f}")
    print(f"  Best Trial: {study.best_trial.number}")
    print(f"  Completed Trials: {len(study.trials)}")
    print(f"  Best Params: {study.best_params}")

# No Compression:
#  Best Accuracy: 0.8352
#  Best Trial: 13
#  Completed Trials: 20
#  Best Params: {'num_layers': 1, 'num_heads': 2, 'hidden_size': 4, 'intermediate_size': 3, 'global_layer_type': 0}

# Compression w/o Post-Training (PTQ):
#  Best Accuracy: 0.8011
#  Best Trial: 13
#  Completed Trials: 20
#  Best Params: {'num_layers': 1, 'num_heads': 2, 'hidden_size': 4, 'intermediate_size': 3, 'global_layer_type': 0}

# Compression w/ Post-Training (QAT):
#  Best Accuracy: 0.8726
#  Best Trial: 18
#  Completed Trials: 20
#  Best Params: {'num_layers': 1, 'num_heads': 2, 'hidden_size': 3, 'intermediate_size': 0, 'global_layer_type': 0}