import torch.nn as nn
from chop.nn.modules import Identity
from chop.tools import get_tokenized_dataset
from chop.tools.utils import deepsetattr
from transformers import AutoConfig, AutoModelForSequenceClassification
import optuna
from chop.tools import get_trainer
from pathlib import Path
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

# Writing a model constructor
def construct_model(trial):
    config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")

    # Update these parameters in the config of pretrained model
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        options = search_space[param] # Search for the Parameter
        chosen_idx = trial.suggest_int(param, 0, len(options) - 1) # Place an index for the chosen parameter
        setattr(config, param, options[chosen_idx]) # Choose the parameter for the chosen index

    # Trial out with the chosen parameters
    trial_model = AutoModelForSequenceClassification.from_config(config)

    # We look at every Linear layer and ask: "Should this be a real layer or a ghost layer (Identity)?"
    for name, layer in trial_model.named_modules():
        if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )

            if new_layer_cls == Identity:
                # Swap the real layer for an Identity (pass-through - Do nothing) layer
                new_layer = Identity()
                deepsetattr(trial_model, name, new_layer)

    return trial_model

# Define the Objective Function
# In each trial create a new model with chosen hyperparameters
def objective(trial):
    model = construct_model(trial)

    # Train it for one epoch
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

    # Save the model inside the trial object so we can retrieve the winner later
    trial.set_user_attr("model", model)

    return accuracy

# Launch the Search
# GridSampler - To iterate through every possibility
# RandomSampler - Randomly choose combination of hyperparameters
# TPESampler - Uses Tree-structured Parzen Estimator algorithm
sampler = optuna.samplers.RandomSampler()
study = optuna.create_study(
    direction="maximize",
    study_name="bert-tiny-nas-study",
    sampler=sampler,
)

# n_trials=1 means we only try ONE randomly picked model
study.optimize(objective, n_trials=1)

print(f"Best Trial Accuracy: {study.best_value}")
print(f"Best Parameters: {study.best_params}")

# Best Trial Accuracy: 0.7798

best_model = study.best_trial.user_attrs["model"].cpu()
save_file = Path(__file__).parent / "Tutorial_5_Best_Model.pkl"

with open(save_file, "wb") as f:
    dill.dump(best_model, f)