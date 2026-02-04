import torch
import torch.nn as nn
from chop.nn.modules import Identity
from chop.tools import get_tokenized_dataset, get_trainer
from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools.utils import deepsetattr
import optuna
from optuna.samplers import GridSampler, RandomSampler, TPESampler
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import dill


dataset, tokenizer = get_tokenized_dataset(
    dataset="imdb",
    checkpoint="bert-base-uncased",
    return_tokenizer=True,
)

# Define the Search Space
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choices": [
        nn.Linear,
        Identity, # "Identity" means "do nothing" (effectively deleting the layer)
    ], 
}

def construct_model(trial):
    config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
    
    # Update these parameters in the config of pretrained model
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        options = search_space[param] # Search for the Parameter
        chosen_idx = trial.suggest_int(param, 0, len(options) - 1) # Place an index for the chosen parameter
        setattr(config, param, options[chosen_idx]) # Choose the parameter for the chosen index
    
    # Build model
    model = AutoModelForSequenceClassification.from_config(config)
    model.config.problem_type = "single_label_classification"

    # We look at every Linear layer and ask: "Should this be a real layer or a ghost layer (Identity)?" 
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )

            if new_layer_cls == Identity:
                # Swap the real layer for an Identity (pass-through - Do nothing) layer
                new_layer = Identity()
                deepsetattr(model, name, new_layer)
    
    return model

def construct_model_for_Grid(trial):
    config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
    
    # Update these parameters in the config of pretrained model
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        options = search_space[param] # Search for the Parameter
        chosen_idx = trial.suggest_int(param, 0, len(options) - 1) # Place an index for the chosen parameter
        setattr(config, param, options[chosen_idx]) # Choose the parameter for the chosen index
    
    # Build model
    model = AutoModelForSequenceClassification.from_config(config)
    model.config.problem_type = "single_label_classification"

    layer_type_choice = trial.suggest_int("global_layer_type", 0, 1)

    if layer_type_choice == 1:
        for name, layer in model.named_modules():
            # Check if swapping is mathematically valid (in_features == out_features)
            if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
                deepsetattr(model, name, Identity())
    
    return model

# Define the Objective Function
# In each trial create a new model with chosen hyperparameters
def objective(trial):
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
    trial.set_user_attr("model", model)

    return accuracy

def objective_for_Grid(trial):
    model = construct_model_for_Grid(trial)

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
    trial.set_user_attr("model", model)

    return accuracy

def run_study(sampler_name, sampler, n_trials=20):
    study = optuna.create_study(
        direction="maximize",
        study_name=f"bert-nas-{sampler_name}",
        sampler=sampler,
    )
    
    study.optimize(objective, n_trials=n_trials)
    return study

def run_study_for_grid(sampler_name, sampler, n_trials=20):
    study = optuna.create_study(
        direction="maximize",
        study_name=f"bert-nas-{sampler_name}",
        sampler=sampler,
    )
    
    study.optimize(objective_for_Grid, n_trials=n_trials)
    return study

n_trials = 20
results = {}

print("\nRANDOM SAMPLER:")
random_study = run_study(
    "random", 
    optuna.samplers.RandomSampler(),
    n_trials=n_trials
)

values = [t.value for t in random_study.trials if t.value is not None]
results["Random"] = np.maximum.accumulate(values)
print(f"Random Best: {max(values):.4f}\n")

print("\nTPE SAMPLER:")
tpe_study = run_study(
    "tpe",
    optuna.samplers.TPESampler(),
    n_trials=n_trials
)

values = [t.value for t in tpe_study.trials if t.value is not None]
results["TPE"] = np.maximum.accumulate(values)
print(f"TPE Best: {max(values):.4f}\n")

print("\nGRID SAMPLER:")
grid_study = run_study_for_grid(
    "grid",
    optuna.samplers.GridSampler(search_space={
        "num_layers": [0, 1, 2],
        "num_heads": [0, 1, 2, 3],
        "hidden_size": [0, 1, 2, 3, 4],
        "intermediate_size": [0, 1, 2, 3, 4],
        "global_layer_type": [0, 1]
    }
    ),
    n_trials=n_trials
)

values = [t.value for t in grid_study.trials if t.value is not None]
results["Grid"] = np.maximum.accumulate(values)
print(f"Grid Best: {max(values):.4f}\n")

plt.figure(figsize=(10, 6))

colors = {"Random": "blue", "TPE": "red", "Grid": "green"}
markers = {"Random": "o", "TPE": "s", "Grid": "^"}

for sampler_name, cumulative_max in results.items():
    trial_numbers = list(range(1, len(cumulative_max) + 1))
    plt.plot(
        trial_numbers, 
        cumulative_max, 
        marker=markers[sampler_name],
        label=f"{sampler_name}Sampler",
        color=colors[sampler_name],
        linewidth=2,
        markersize=6
    )

plt.xlabel("Number of Trials", fontsize=13)
plt.ylabel("Maximum Achieved Accuracy", fontsize=13)
plt.title("Performance Comparison of Optuna Samplers", fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save the figure
save_path = Path(__file__).parent / "Lab_2_Sampler_Comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print("\n\nSUMMARY:")

studies = {
    "Random": random_study,
    "TPE": tpe_study,
    "Grid": grid_study
}

for name, study in studies.items():
    print(f"\n{name}Sampler:")
    print(f"  Best Accuracy: {study.best_value:.4f}")
    print(f"  Best Trial Number: {study.best_trial.number}")
    print(f"  Number of Completed Trials: {len(study.trials)}")

# Save the best model
best_overall = max(studies.items(), key=lambda x: x[1].best_value)
print(f"\n{'=' * 60}")
print(f"OVERALL WINNER: {best_overall[0]}Sampler")
print(f"Best Accuracy: {best_overall[1].best_value:.4f}")
print(f"{'=' * 60}")

best_model = best_overall[1].best_trial.user_attrs["model"].cpu()
model_path = Path(__file__).parent / "Lab_2_Task_1_Best_Model.pkl"
with open(model_path, "wb") as f:
    dill.dump(best_model, f)
print(f"\nBest model saved to: {model_path}")

#SUMMARY:

#RandomSampler:
#Best Accuracy: 0.8315
#Best Trial Number: 18
#Number of Completed Trials: 20

#TPESampler:
#Best Accuracy: 0.8339
#Best Trial Number: 19
#Number of Completed Trials: 20

#GridSampler:
#Best Accuracy: 0.8352
#Best Trial Number: 13
#Number of Completed Trials: 20